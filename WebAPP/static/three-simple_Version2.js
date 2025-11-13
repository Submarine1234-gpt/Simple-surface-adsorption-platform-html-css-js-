// 简化的Three.js功能替代方案
// 这是一个轻量级的3D渲染解决方案

window.THREE = (function() {
    'use strict';
    
    // 基础向量3D类
    function Vector3(x, y, z) {
        this.x = x || 0;
        this.y = y || 0;
        this.z = z || 0;
    }
    
    Vector3.prototype.set = function(x, y, z) {
        this.x = x;
        this.y = y;
        this.z = z;
        return this;
    };
    
    // 颜色类
    function Color(r, g, b) {
        this.r = r !== undefined ? r : 1;
        this.g = g !== undefined ? g : 1;
        this.b = b !== undefined ? b : 1;
    }
    
    Color.prototype.setRGB = function(r, g, b) {
        this.r = r;
        this.g = g;
        this.b = b;
        return this;
    };
    
    // 场景类
    function Scene() {
        this.children = [];
        this.background = null;
    }
    
    Scene.prototype.add = function(object) {
        this.children.push(object);
    };
    
    // 相机类
    function PerspectiveCamera(fov, aspect, near, far) {
        this.fov = fov || 50;
        this.aspect = aspect || 1;
        this.near = near || 0.1;
        this.far = far || 2000;
        this.position = new Vector3();
    }
    
    PerspectiveCamera.prototype.lookAt = function(target) {
        // 简化的lookAt实现
        this.target = target;
    };
    
    PerspectiveCamera.prototype.updateProjectionMatrix = function() {
        // 更新投影矩阵（简化）
    };
    
    // 渲染器类
    function WebGLRenderer(options) {
        options = options || {};
        this.domElement = document.createElement('canvas');
        this.context = this.domElement.getContext('2d');
        this.width = 800;
        this.height = 600;
    }
    
    WebGLRenderer.prototype.setSize = function(width, height) {
        this.width = width;
        this.height = height;
        this.domElement.width = width;
        this.domElement.height = height;
        this.domElement.style.width = width + 'px';
        this.domElement.style.height = height + 'px';
    };
    
    WebGLRenderer.prototype.render = function(scene, camera) {
        const ctx = this.context;
        ctx.clearRect(0, 0, this.width, this.height);
        
        // 绘制背景
        if (scene.background) {
            ctx.fillStyle = `rgb(${Math.floor(scene.background.r * 255)}, ${Math.floor(scene.background.g * 255)}, ${Math.floor(scene.background.b * 255)})`;
            ctx.fillRect(0, 0, this.width, this.height);
        }
        
        // 简化的3D到2D投影
        scene.children.forEach(child => {
            if (child.geometry && child.material) {
                this.renderMesh(child, camera, ctx);
            }
        });
    };
    
    WebGLRenderer.prototype.renderMesh = function(mesh, camera, ctx) {
        // 简化的网格渲染
        if (mesh.geometry.type === 'SphereGeometry') {
            // 渲染球体
            const x = this.width/2 + mesh.position.x * 10;
            const y = this.height/2 - mesh.position.y * 10;
            const radius = mesh.geometry.radius * 10;
            
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            if (mesh.material.color) {
                ctx.fillStyle = `rgb(${Math.floor(mesh.material.color.r * 255)}, ${Math.floor(mesh.material.color.g * 255)}, ${Math.floor(mesh.material.color.b * 255)})`;
            } else {
                ctx.fillStyle = 'gray';
            }
            ctx.fill();
        }
    };
    
    WebGLRenderer.prototype.dispose = function() {
        // 清理资源
    };
    
    // 几何体类
    function BufferGeometry() {
        this.attributes = {};
        this.type = 'BufferGeometry';
    }
    
    BufferGeometry.prototype.setAttribute = function(name, attribute) {
        this.attributes[name] = attribute;
    };
    
    BufferGeometry.prototype.computeVertexNormals = function() {
        // 计算顶点法线（简化）
    };
    
    function SphereGeometry(radius, widthSegments, heightSegments) {
        BufferGeometry.call(this);
        this.type = 'SphereGeometry';
        this.radius = radius || 1;
    }
    
    SphereGeometry.prototype = Object.create(BufferGeometry.prototype);
    
    // 材质类
    function MeshLambertMaterial(options) {
        options = options || {};
        this.color = options.color || new Color(1, 1, 1);
        this.vertexColors = options.vertexColors || false;
        this.side = options.side || 0;
        this.transparent = options.transparent || false;
        this.opacity = options.opacity !== undefined ? options.opacity : 1;
    }
    
    // 网格类
    function Mesh(geometry, material) {
        this.geometry = geometry;
        this.material = material;
        this.position = new Vector3();
        this.rotation = new Vector3();
    }
    
    // 光照类
    function AmbientLight(color, intensity) {
        this.color = color || 0x404040;
        this.intensity = intensity !== undefined ? intensity : 1;
    }
    
    function DirectionalLight(color, intensity) {
        this.color = color || 0xffffff;
        this.intensity = intensity !== undefined ? intensity : 1;
        this.position = new Vector3();
    }
    
    DirectionalLight.prototype.set = function(x, y, z) {
        this.position.set(x, y, z);
    };
    
    // 工具类
    function Box3() {
        this.min = new Vector3();
        this.max = new Vector3();
    }
    
    Box3.prototype.setFromObject = function(object) {
        // 简化的包围盒计算
        this.min.set(-10, -10, -10);
        this.max.set(10, 10, 10);
        return this;
    };
    
    Box3.prototype.getCenter = function(target) {
        return target.set(
            (this.min.x + this.max.x) * 0.5,
            (this.min.y + this.max.y) * 0.5,
            (this.min.z + this.max.z) * 0.5
        );
    };
    
    Box3.prototype.getSize = function(target) {
        return target.set(
            this.max.x - this.min.x,
            this.max.y - this.min.y,
            this.max.z - this.min.z
        );
    };
    
    function Float32Array(data) {
        return new window.Float32Array(data);
    }
    
    // 返回简化的THREE对象
    return {
        Scene: Scene,
        PerspectiveCamera: PerspectiveCamera,
        WebGLRenderer: WebGLRenderer,
        BufferGeometry: BufferGeometry,
        SphereGeometry: SphereGeometry,
        MeshLambertMaterial: MeshLambertMaterial,
        Mesh: Mesh,
        AmbientLight: AmbientLight,
        DirectionalLight: DirectionalLight,
        Vector3: Vector3,
        Color: Color,
        Box3: Box3,
        Float32Array: Float32Array,
        DoubleSide: 2
    };
})();

// 添加requestAnimationFrame兼容性
if (!window.requestAnimationFrame) {
    window.requestAnimationFrame = function(callback) {
        return window.setTimeout(callback, 1000 / 60);
    };
}