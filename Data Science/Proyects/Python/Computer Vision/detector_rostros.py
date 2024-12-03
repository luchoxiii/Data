from matplotlib import pyplot
from matplotlib.patches import Rectangle,Circle
from mtcnn.mtcnn import MTCNN
import cv2


## Some of Us by Starsailor

#Vamos a capturar el rostro

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    cv2.imshow("Rostro",frame)
    if   cv2.waitKey(1) == 27:
        break

cv2.imwrite("Rostro.jpg", frame)
cap.release()
cv2.destroyAllWindows()

#Creamos una funcion que nos permita dibujar rectangulos y circulos

def dibujo(img, lista_resultados):
    # Mostrar caras
    print(lista_resultados)

    # Cargar la imagen
    imagen = pyplot.imread(img)

    # Mostrar la imagen
    pyplot.imshow(imagen)

    # Obtener los ejes para dibujar
    ax = pyplot.gca()

    # Dibujar las cajas alrededor de las caras y los puntos clave
    for result in lista_resultados:
        x, y, ancho, alto = result["box"]
        rect = Rectangle((x, y), ancho, alto, fill=False, color="green")
        ax.add_patch(rect)

        for puntos, value in result["keypoints"].items():
            dot = Circle(value, radius=4, color="green")
            ax.add_patch(dot)

    # Crear subgráficas de las caras detectadas
    for i in range(len(lista_resultados)):
        x1, y1, ancho1, alto1 = lista_resultados[i]["box"]
        x2, y2 = x1 + ancho1, y1 + alto1

        pyplot.subplot(1, len(lista_resultados), i+1)
        pyplot.axis("off")
        pyplot.imshow(imagen[y1:y2, x1:x2])

    # Mostrar la imagen con las caras detectadas y las subgráficas
    pyplot.show()

# Ruta de la imagen
img = "Rostro.jpg"

# Cargar la imagen para detectar caras
pixeles = pyplot.imread(img)

# Crear el detector de caras
detector = MTCNN()

# Detectar las caras en la imagen
caras = detector.detect_faces(pixeles)

# Llamar a la función de dibujo para mostrar resultados
dibujo(img, caras)