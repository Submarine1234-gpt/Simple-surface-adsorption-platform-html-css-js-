import os
import json
import time
import uuid
import logging
import threading
import zipfile
import subprocess
import sys
from datetime import datetime
from collections import deque
from flask import Flask, request, jsonify, render_template, send_file, Response

# ========== 配置部分 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'user_uploads')  # 用户上传文件主目录，绝对路径
LOG_DIR = os.path.join(BASE_DIR, 'logs')
MAX_RESULTS_PER_USER = 10       # 每个用户最多保留多少条历史记录

# 日志文件配置
os.makedirs(LOG_DIR, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(LOG_DIR, 'app.log'), encoding='utf-8')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)
app.logger.addHandler(file_handler)

# 线程锁和用户历史结果
results_lock = threading.Lock()
user_results = {}  # {session_id: deque([...])}

# ========== 工具函数 ==========
def get_user_path(subfolder, session_id=None):
    """直接根据类型和session生成目录，去掉IP隔离"""
    base = os.path.join(UPLOAD_FOLDER)
    return os.path.join(base, subfolder, session_id) if session_id else os.path.join(base, subfolder)

def run_core_py(params, log_file):
    """调用 core.py 脚本并实时记录日志"""
    python = sys.executable
    core_py = os.path.join(BASE_DIR, 'core.py')
    cmd = [python, core_py]
    # 参数转命令行
    for k, v in params.items():
        if isinstance(v, bool):
            if v: cmd.append(f'--{k}')
        elif v is not None:
            cmd.extend([f'--{k}', str(v)])
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    app.logger.info(f"Running: {' '.join(cmd)}")
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[SYSTEM] Executing: {' '.join(cmd)}\n\n"); f.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1, env=env)
        for line in iter(proc.stdout.readline, ''):
            f.write(line); f.flush()
        proc.stdout.close()
        if proc.wait() != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

# ========== 路由：提交计算任务 ==========
@app.route('/run-calculation', methods=['POST'])
def run_calculation():
    session_id = str(uuid.uuid4())
    result_dir = get_user_path('results', session_id)
    os.makedirs(result_dir, exist_ok=True)
    log_file = os.path.join(result_dir, 'calculation.log')
    try:
        # 1. 保存上传的文件
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"[SYSTEM] Calculation session starting: {datetime.now()}\n")
            f.write(f"[SYSTEM] Session ID: {session_id}\n\n")
        print("request.files:", request.files)
        print("request.form:", request.form)
        substrate = request.files.get('substrate_file')
        adsorbate = request.files.get('adsorbate_file')
        if not substrate or not adsorbate:
            raise ValueError("Both CIF files must be uploaded.")
        substrate_path = os.path.join(result_dir, substrate.filename)
        adsorbate_path = os.path.join(result_dir, adsorbate.filename)
        try:
            substrate.save(substrate_path)
            adsorbate.save(adsorbate_path)
            print(f"Saved substrate to {substrate_path}")
            print(f"Saved adsorbate to {adsorbate_path}")
        except Exception as e:
            print(f"Error saving files: {e}")
            raise
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[SYSTEM] Saved substrate to: {substrate_path}\n")
            f.write(f"[SYSTEM] Saved adsorbate to: {adsorbate_path}\n\n")
            f.write(f"[SYSTEM] Files in result_dir: {os.listdir(result_dir)}\n\n")

        # 2. 组装参数
        params = {k: request.form.get(k) for k in request.form}
        params.update({
            'substrate': substrate_path,
            'adsorbate': adsorbate_path,
            'output_folder': result_dir,
            'hollow_sites_enabled': 'hollow_sites_enabled' in request.form,
            'on_top_sites_enabled': 'on_top_sites_enabled' in request.form,
            'place_on_bottom': 'place_on_bottom' in request.form,
            'rotation_method': 'rotation_method' in request.form,
        })
        surface_axis = request.form.get('surface_axis', '2')

        # 3. 后台线程执行主计算
        def background_task(sid, outdir, axis):
            try:
                run_core_py(params, log_file)
                # 打包结果
                zip_dir = get_user_path('zips')
                os.makedirs(zip_dir, exist_ok=True)
                zip_name = f"results_{sid}.zip"
                zip_path = os.path.join(zip_dir, zip_name)
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, _, files in os.walk(outdir):
                        for file in files:
                            if not file.endswith('.done'):
                                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), outdir))
                # 记录历史
                with results_lock:
                    if sid not in user_results:
                        user_results[sid] = deque(maxlen=MAX_RESULTS_PER_USER + 1)
                    user_results[sid].append({
                        'session_id': sid,
                        'filename': zip_name,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'surface_axis': axis
                    })
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write("\n[SYSTEM] All files zipped successfully.\n")
            except Exception as e:
                app.logger.error(f"后台任务失败: {e}", exc_info=True)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n[ERROR] Background task failed: {str(e)}\n")
            finally:
                with open(os.path.join(outdir, '.done'), 'w') as f:
                    f.write('done')

        threading.Thread(target=background_task, args=(session_id, result_dir, surface_axis)).start()
        return jsonify({'success': True, 'session_id': session_id, 'surface_axis': surface_axis})
    except Exception as e:
        app.logger.error(f"初始化失败: {e}", exc_info=True)
        return jsonify({'success': False, 'message': str(e)}), 400

# --- 其它路由 ---
@app.route('/check-status/<session_id>')
def check_status(session_id):
    done_file_path = os.path.join(get_user_path('results', session_id), '.done')
    return jsonify({'status': 'complete' if os.path.exists(done_file_path) else 'running'})

@app.route('/stream-logs/<session_id>')
def stream_logs(session_id):
    log_file = os.path.join(get_user_path('results', session_id), 'calculation.log')
    def generate():
        if not os.path.exists(log_file):
            yield "data: [SYSTEM] Waiting for log file to be created...\n\n"; time.sleep(1)
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                line = f.readline()
                if not line:
                    done_file = os.path.join(get_user_path('results', session_id), '.done')
                    if os.path.exists(done_file):
                        yield "data: [SYSTEM] Log stream finished.\n\n"; break
                    time.sleep(0.2); continue
                yield f"data: {line.strip()}\n\n"
    return Response(generate(), content_type='text/event-stream; charset=utf-8')

@app.route('/get-viz-data/<session_id>/<filename>')
def get_viz_data(session_id, filename):
    if filename not in ['adsorption_sites.json', 'surface_atoms.json']:
        return jsonify({"error": "Invalid data file requested"}), 400
    file_path = os.path.join(get_user_path('results', session_id), filename)
    try:
        if not os.path.exists(file_path):
            return jsonify({"error": f"{filename} not available or still generating."}), 404
        return send_file(file_path)
    except Exception as e:
        app.logger.error(f"Failed to serve {file_path}: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/get-results')
def get_results():
    with results_lock:
        # 所有历史都用 session_id 做key，拿最新
        all_results = []
        for dq in user_results.values():
            all_results.extend(list(dq))
        return jsonify(list(reversed(all_results)))

@app.route('/download-result/<session_id>')
def download_result(session_id):
    zip_path = os.path.join(get_user_path('zips'), f"results_{session_id}.zip")
    if not os.path.exists(zip_path): return "Result file not found.", 404
    return send_file(zip_path, as_attachment=True)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)