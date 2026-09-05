import cv2
import numpy as np
import math

img = cv2.imread("triage.jpeg")
h, w = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
is_black = gray < 40

b =img[:,:, 0]
g =img[:,:, 1]
r =img[:, :, 2]
is_blue = (b > 110) & (r < 130) & (g < 100)

canwalk = np.ones((h, w), dtype=np.uint8)
canwalk[is_black | is_blue] = 0

orangepixels = np.where((r >= 235) & (g >= 95) & (g <= 135) & (b <= 50))
start = (int(np.mean(orangepixels[1])), int(np.mean(orangepixels[0])))

purplepixel = np.where((r > 170) & (g > 80) & (g < 140) & (b > 190))
goal = (int(np.mean(purplepixel[1])), int(np.mean(purplepixel[0])))

canwalk[start[1], start[0]] = 1
canwalk[goal[1], goal[0]] = 1

open_list = [(math.hypot(start[0] -goal[0], start[1]- goal[1]), start[0],start[1])]
camefrom = {}
g_score = {start: 0}

moves = [(-1,0), (1,0), (0,-1),(0,1), (-1,-1), (-1,1),(1,-1), (1,1)]
while len(open_list) > 0:

    open_list.sort(key=lambda item: item[0])
    _, x, y = open_list.pop(0)

    if (x, y) == goal:
        break
    for dx, dy in moves:
        nx, ny = x + dx, y + dy

        if 0 <= nx < w and 0 <= ny < h and canwalk[ny, nx] == 1:
            step_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
            new_g = g_score[(x, y)] + step_cost

            if new_g < g_score.get((nx, ny), 999999999):
                camefrom[(nx, ny)] = (x, y)
                g_score[(nx, ny)] = new_g
                h_cost = math.hypot(nx - goal[0], ny - goal[1])
                open_list.append((new_g + h_cost, nx, ny))

curr = goal
while curr in camefrom:
    prev = camefrom[curr]
    cv2.line(img, prev, curr, (100, 50, 40), 3) 
    curr = prev

cv2.imwrite("done.png",img)
cv2.imshow("done",img)
cv2.waitKey(0)
cv2.destroyAllWindows()