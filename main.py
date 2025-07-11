import math

import cv2
import numpy as np

# 预处理，提取边缘与轮廓
image_path = "target1.png"
image = cv2.imread(image_path)  # 读取图像
blurred = cv2.GaussianBlur(image, (19, 19), 0)  # 高斯模糊去噪，核大小调整
edges = cv2.Canny(blurred, 30, 100)  # Canny边缘检测，阈值调整
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
cv2.imwrite(image_path + '_output1.jpg', edges)  # 生成调试用边缘图像

# 拟合并取得最大面积的椭圆
outer_ellipse = [None, None]
max_area = [0, 0]
for cnt in contours:
    if len(cnt) >= 100:
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (width, height), angle = ellipse
        aspect_ratio = min(width, height) / max(width, height)
        if aspect_ratio > 0.7 and width > 100:  # 排除过扁或过小的椭圆
            area = np.pi * (width / 2) * (height / 2)
            if area > max_area[0]:  # 取面积最大的椭圆
                max_area[0] = area
                outer_ellipse[0] = ellipse

# 拟合并取得第二大面积且圆心相近的椭圆
for cnt in contours:
    if len(cnt) >= 100:
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (width, height), angle = ellipse

        aspect_ratio = min(width, height) / max(width, height)
        if aspect_ratio > 0.7 and width > 100:  # 排除过扁或过小的椭圆
            (x, y), (a, b), angle = outer_ellipse[0]
            area = np.pi * (width / 2) * (height / 2)
            if max_area[1] < area < max_area[0] * 0.8 and math.dist([cx, cy], [x, y]) < 20:
                max_area[1] = area
                outer_ellipse[1] = ellipse

(cx, cy), (a, b), angle = outer_ellipse[0]
(cx1, cy1), (a1, b1), angle1 = outer_ellipse[1]

# 计算靶心所在位置
center_x = cx + (cx1 - cx) * math.pow((max_area[0] / max_area[1]), 0.7)  # 校正靶心坐标
center_y = cy + (cy1 - cy) * math.pow((max_area[0] / max_area[1]), 0.7)

# 输出调试图像
debug_img = image.copy()
cv2.ellipse(debug_img, outer_ellipse[0], (0, 255, 0), 5)  # 绘制椭圆轮廓
cv2.ellipse(debug_img, outer_ellipse[1], (0, 255, 0), 5)
cv2.circle(debug_img, (int(center_x), int(center_y)), 3, (0, 0, 255), 5)  # 绘制靶心
cv2.circle(debug_img, (int(cx), int(cy)), 3, (0, 255, 0), 5)
cv2.circle(debug_img, (int(cx1), int(cy1)), 3, (0, 255, 0), 5)
cv2.imwrite(image_path + '_output2.jpg', debug_img)  # 生成调试用椭圆图像

# 计算透视变换：原图像椭圆端点
theta = np.deg2rad(angle)
cos_theta, sin_theta = np.cos(theta), np.sin(theta)
src_pt = []
src_pt.append((cx + a / 2 * cos_theta, cy + a / 2 * sin_theta))  # 长轴端点
src_pt.append((cx - b / 2 * sin_theta, cy + b / 2 * cos_theta))  # 短轴端点
src_pt.append((cx - a / 2 * cos_theta, cy - a / 2 * sin_theta))
src_pt.append((cx + b / 2 * sin_theta, cy - b / 2 * cos_theta))
src_pts = np.float32(src_pt)  # 取得原图像椭圆的四个端点

# 计算透视变换：目标图像正圆对应点
r = (a + b) / 4
dst_pt = []
for i in range(4):
    center = np.array((center_x, center_y))
    point = np.array(src_pt[i])
    vec_to_point = point - center
    dist = np.linalg.norm(vec_to_point)
    unit_vector = vec_to_point / dist
    dst_pt.append((center + r * unit_vector))
dst_pts = np.float32(dst_pt)

M = cv2.getPerspectiveTransform(src_pts, dst_pts, cv2.DECOMP_LU)  # 计算矩阵

# 输出调试图像
debug_img = image.copy()
cv2.ellipse(debug_img, outer_ellipse[0], (0, 255, 0), 5)  # 绘制椭圆轮廓
cv2.ellipse(debug_img, outer_ellipse[1], (0, 255, 0), 5)
cv2.circle(debug_img, (int(center_x), int(center_y)), 3, (0, 0, 255), 5)  # 绘制靶心
for pt in src_pt:
    cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), 5)
for pt in dst_pt:
    cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), 5)
cv2.imwrite(image_path + '_output3.jpg', debug_img)  # 生成调试用椭圆图像

# 应用透视变换
h, w = image.shape[:2]
transformed = cv2.warpPerspective(image, M, (w, h))

# 输出调试图像
debug_img = transformed.copy()
cv2.circle(debug_img, (int(center_x), int(center_y)), 3, (0, 255, 0), 5)  # 绘制靶心
for length in [1, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6, 1 / 12]:
    cv2.circle(debug_img, (int(center_x), int(center_y)), int(r * length), (0, 255, 0), 5)  # 绘制正圆轮廓
for pt in dst_pt:
    cv2.circle(debug_img, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), 5)
cv2.imwrite(image_path + '_output4.jpg', debug_img)  # 生成最终图像
