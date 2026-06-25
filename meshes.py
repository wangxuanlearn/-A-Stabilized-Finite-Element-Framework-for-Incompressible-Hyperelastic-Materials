from dolfin import *  

def create_cooks_mesh(nx=12, ny=6):
    """
    Cook's membrane 几何
    四个顶点:
        (0, 0), (0, 4.4), (4.8, 4.4), (4.8, 6.0)
    上边界水平 (Y=4.4)，下边界从 0 到 6.0
    """
    L=4.8                        
    H_total = 6.0    # 用于映射的参考高度
    
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

def create_anisotropic_compressblock_mesh(nx=20, ny=10, scale=1):
    """
    矩形块网格 (用于不可压缩验证)
    
    几何 (默认单位: cm):
        总宽: 20 cm, 总高: 10 cm
        四个顶点:
            A (0, 0), B (0, 10), C (20, 0), D (20, 10)
    
    边界条件:
        左侧 (x=0): u_x = 0
        底部 (y=0): u_y = 0
        顶部 (y=10): 受均匀压力载荷
        右侧 (x=20): 自由
    
    参数:
        nx, ny: 网格划分密度
        scale: 几何缩放因子 (如果 scale=0.01，则转为 m 单位)
    
    返回:
        mesh: 网格对象
        boundaries: 边界标记
            (1: 左边界, 2: 底部边界, 3: 顶部边界, 4: 右边界)
    """
    # 几何尺寸 (默认 cm)
    L = 20.0 * scale      # 总宽度
    H = 10.0 * scale      # 总高度
    
    # 1. 创建矩形网格
    mesh = RectangleMesh(Point(0.0, 0.0), Point(L, H), nx, ny)
    
    # 左边界 (x ≈ 0) → 标记为 1
    class LeftBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], 0.0)
        
    # 底部边界 (y ≈ 0) → 标记为 2
    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 0.0)
    
    # 顶部边界 (y ≈ H) → 标记为 3
    class TopBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], H)
    
    # 右侧边界 (x ≈ L) → 标记为 4
    class RightBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[0], L)
    
    # 在上边界 [5,15] 范围标记为新边界 5，其余顶部保持标记 3
    class LoadedTop(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], H) and x[0] >= 5.0 and x[0] <= 15.0
    
    # 创建边界标记
    boundaries = MeshFunction("size_t", mesh, mesh.geometry().dim() - 1)
    boundaries.set_all(0)
    
    # 标记边界
    LeftBoundary().mark(boundaries, 1)
    BottomBoundary().mark(boundaries, 2)
    TopBoundary().mark(boundaries, 3)
    RightBoundary().mark(boundaries, 4)
    LoadedTop().mark(boundaries, 5)
    
    return mesh, boundaries