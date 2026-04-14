import cv2

def mostra_hsv(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"HSV em ({x},{y}): {param[y,x]}")

img = cv2.imread(r"C:\Users\pedro\Documents\INSPER\SEM_07\VISAO\Projeto1\_Eucalipto_Escolhidos1\Eucalipto5.jpg")
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("Clique na imagem", img)
cv2.setMouseCallback("Clique na imagem", mostra_hsv, img_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Folhas:
# HSV em (1240,1894): [ 37 153 183]
# HSV em (1425,1877): [ 40 129 138]
# HSV em (1303,2159): [ 44 105  97]
# HSV em (1418,1782): [ 31  74 172]
# HSV em (1335,1812): [175 147 146]
# HSV em (1345,1781): [ 39 106 159]

# Caule:
# HSV em (1480,2333): [ 35 151 133]
# HSV em (1396,2029): [ 34 189 165]
# HSV em (1352,1851): [ 21  54 128]
# HSV em (1525,2507): [177  48  48]

# vaso:
# HSV em (1562,2542): [109  81  66]
# HSV em (1452,2593): [110  57  81]
# HSV em (1650,2703): [108 106  77]
# HSV em (1476,2729): [110   9  87]

# Base:
# HSV em (1596,2937): [ 26  18 200]
# HSV em (1449,2894): [ 26  17 213]
# HSV em (1504,3209): [103   9 207]
# HSV em (1732,3171): [106  17 169]
# HSV em (1689,3068): [ 36   7 177]

# Colete:
# HSV em (1522,2488): [ 16 116  97]
# HSV em (1525,2499): [150  39  46]
# HSV em (1514,2603): [ 9 75 82]
# HSV em (1525,2510): [ 5 62 49]
# HSV em (1526,2519): [ 5 60 55]
# HSV em (1526,2528): [ 5 61 50]