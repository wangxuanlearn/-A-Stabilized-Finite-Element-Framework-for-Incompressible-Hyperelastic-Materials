from dolfin import *  

def create_cooks_mesh(nx=12, ny=6):
    """
    Cook's membrane 几何
    四个顶点:
        (0, 0), (0, 44), (48, 44), (48, 60)
    上边界水平 (Y=44)，下边界从 0 到 60
    """
    L = 48.0
    H_total = 60.0      # 用于映射的参考高度
    
    # 上边界在 x 位置的 y 坐标: 从 (0,44) 到 (48,60)
    top_start = 44.0    # x=0 时上边界 y=44
    top_end = 60.0      # x=48 时上边界 y=60
    
    # 下边界在 x 位置的 y 坐标: 从 (0,0) 到 (48,44)
    bottom_start = 0.0  # x=0 时下边界 y=0
    bottom_end = 44.0   # x=48 时下边界 y=44
    
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



