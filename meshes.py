from dolfin import *  

def create_cooks_mesh(nx=12, ny=6):
    """
    Cook's membrane 几何
    四个顶点:
        (0, 0), (0, 4.4), (4.8, 4.4), (4.8, 6.0)
    上边界水平 (Y=4.4)，下边界从 0 到 6.0
    """
    L = 4.80
    H_total = 6.00      # 用于映射的参考高度
    
    # 上边界在 x 位置的 y 坐标: 从 (0,4.4) 到 (4.8,6.0)
    top_start = 4.4    # x=0 时上边界 y=4.4
    top_end = 6.0      # x=4.8 时上边界 y=6.0
    
    # 下边界在 x 位置的 y 坐标: 从 (0,0) 到 (4.8,4.4)
    bottom_start = 0.0  # x=0 时下边界 y=0
    bottom_end = 4.4   # x=48 时下边界 y=4.4
    
    # 创建矩形网格
    mesh_rect = RectangleMesh(Point(0, 0), Point(L, H_total), nx, ny)
    
    # 变换坐标
    coords = mesh_rect.coordinates()
    for i, coord in enumerate(coords):
        x, y = coord[0], coord[1]
        
        # 上边界: 从 (0,44) 到 (48,60) 的线性插值
        top_y = top_start + (top_end - top_start) * (x / L)
        
        # 下边界: 从 (0,0) 到 (48,44) 的线性插值
        bottom_y = bottom_start + (bottom_end - bottom_start) * (x / L)
        
        # 将 y 从 [0, H_total] 映射到 [bottom_y, top_y]
        coords[i, 1] = bottom_y + y * (top_y - bottom_y) / H_total
    
    # ===== 添加边界标记 =====
    # 创建边界网格函数（用于标记边界）
    boundaries = MeshFunction("size_t", mesh_rect, mesh_rect.geometry().dim() - 1)
    boundaries.set_all(0)  # 默认所有边界标记为 0
    
    # 左边界 (X=0) → 标记为 1
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0)
    
    # 右边界 (X=48) → 标记为 2
    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], L)
    
    # 标记边界
    LeftBoundary().mark(boundaries, 1)
    RightBoundary().mark(boundaries, 2)
    
    return mesh_rect, boundaries

def create_three_dimensional_beam_mesh(mesh_size=0.25):
    """
    创建三维梁的网格和边界条件
    
    参数:
        L: 梁长度 (x方向)
        W: 梁宽度 (y方向)  
        H: 梁高度 (z方向)
        mesh_size: 平均网格尺寸
    
    返回:
        mesh: 网格对象
        bcs: 边界条件列表
        ds: 边界测度 (用于施加载荷)
    """
    L = 6.0   # 长度 (x方向)
    W = 1.0   # 宽度 (y方向)
    H = 1.0   # 高度 (z方向)
        
    # 1. 创建网格
    nx = int(L / mesh_size)
    ny = int(W / mesh_size)
    nz = int(H / mesh_size)
    
    mesh = BoxMesh(Point(0, -W/2, -H/2), Point(L, W/2, H/2), nx, ny, nz)
        
    # ---- 边界 6: X=0 端面 (压力加载) ----
    class LeftEnd(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0)
    
    # ---- 边界 7: X=6 端面 (压力加载) ----
    class RightEnd(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], L)
    
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundaries.set_all(0)
    
    # 标记边界
    LeftEnd().mark(boundaries, 6)
    RightEnd().mark(boundaries, 7)
    
    return mesh, boundaries
