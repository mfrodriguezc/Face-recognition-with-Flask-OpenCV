"""
Proyecto realizado por:
Martin Felipe Rodriguez Caraballo
"""
import cv2
from flask import Flask, render_template, Response, jsonify
import time
import os
import imutils
import numpy as np

app = Flask(__name__)

# Si tienes varias cámaras puedes acceder a ellas en 1, 2, etcétera (en lugar de 0)
camara = cv2.VideoCapture('F:/Proyectos Python/reconocimientofacial1/ElonMusk.mp4')
#camara = cv2.VideoCapture(0)
ruidos=cv2.CascadeClassifier('F:\Proyectos Python\entrenamientos opencv ruidos\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
imgs=[]
rostrosData=[]
ids=[]
bandera = False
# Una función generadora para stremear la cámara
# https://flask.palletsprojects.com/en/1.1.x/patterns/streaming/

def generador_frames():
    while True:
        ok, imagen = obtener_frame_camara()
        if not ok:
            break
        else:
            # Regresar la imagen en modo de respuesta HTTP
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + imagen + b"\r\n"

def obtener_frame_camara():
    ok, frame = camara.read()
    if not ok:
        return False, None
    # Escribir en el array imgs cada frame desde que bandera esta activa
    if bandera:
        imgs.append(frame)
    # Codificar la imagen como JPG
    _, bufer = cv2.imencode(".jpg", frame)
    imagen = bufer.tobytes()
    return True, imagen

# Cuando visiten la ruta
@app.route("/streaming_camara")
def streaming_camara():
    return Response(generador_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Cuando visiten /, servimos el index.html
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/tomar_y_entrenar_fotos")
def entrenamiento_fotos():
    global bandera
    if bandera:
        return jsonify(False)
    bandera = True
    #13 segundos es lo optimo, entre mas segundos mas imagenes para entrenar
    time.sleep(13)
    bandera = False
    longitud = len(imgs)
    #print(longitud)
    modelo='FotosDePrueba'
    ruta1='F:/Proyectos Python/reconocimientofacial/Data'
    rutacompleta = ruta1 + '/'+ modelo
    if not os.path.exists(rutacompleta):
        os.makedirs(rutacompleta)
    
    for i in range(longitud):
        captura=imutils.resize(imgs[i],width=640)
        grises=cv2.cvtColor(captura, cv2.COLOR_BGR2GRAY)
        idcaptura=captura.copy()
        cara=ruidos.detectMultiScale(grises,1.3,5)
        for(x,y,e1,e2) in cara:
            cv2.rectangle(captura, (x,y), (x+e1,y+e2), (0,255,0),2)
            rostrocapturado=idcaptura[y:y+e2,x:x+e1]
            rostrocapturado=cv2.resize(rostrocapturado, (160,160),interpolation=cv2.INTER_CUBIC)
            grisesrostro=cv2.cvtColor(rostrocapturado, cv2.COLOR_BGR2GRAY)
            rostrosData.append(grisesrostro)
            cv2.imwrite(rutacompleta+ '/imagen_{}.jpg'.format(i), rostrocapturado)
            ids.append(0)
    #print('Iniciando el entrenamiento...espere')
    face_recognizer=cv2.face.EigenFaceRecognizer_create()
    face_recognizer.train(rostrosData,np.array(ids))
    face_recognizer.write('EntrenamientoPrueba.xml')
    #print('entrenamiento concluido...')
    return jsonify(True)


@app.route("/reconocer")
def reconocimiento_rostro():
    entrenamientoEigenFaceRecognizer=cv2.face.EigenFaceRecognizer_create()
    var="EntrenamientoPrueba"
    entrenamientoEigenFaceRecognizer.read('{}'.format(var)+'.xml')
    ruidos=cv2.CascadeClassifier('F:\Proyectos Python\entrenamientos opencv ruidos\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
    respuesta,captura=camara.read()
    if not respuesta:
        return False, None
    captura=imutils.resize(captura,width=640)
    grises=cv2.cvtColor(captura, cv2.COLOR_BGR2GRAY)
    idcaptura=grises.copy()
    cara=ruidos.detectMultiScale(grises,1.3,5)
    for(x,y,e1,e2) in cara:
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]
        rostrocapturado=cv2.resize(rostrocapturado, (160,160),interpolation=cv2.INTER_CUBIC)
        resultado=entrenamientoEigenFaceRecognizer.predict(rostrocapturado)
        #se define el umbral para reconocer el rostro, entre mas bajo. mas similar al rostro buscado es.
        if resultado[1]<5700:
            nombre = var
            break
        else:
            nombre = "No encontrado"

    return jsonify({
        "ok": respuesta,
        "nombre_foto": nombre,})



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
