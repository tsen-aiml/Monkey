# import library
from flask import Flask, request, send_file
import os
import cv2
from ultralytics import YOLO
from playsound import playsound

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    video_path = request.json.get('./monkey_in.mp4')
    model_path = request.json.get('./monkey_best.pt')

    output_path = './monkey_out.mp4'

    detect_and_annotate(video_path, output_path, model_path)

    return {'message': 'Detection completed. Check output.'}

@app.route('/output', methods=['GET'])
def get_output():
    output_file = './monkey_out.mp4'
    return send_file(output_file, as_attachment=True) #mimetype='vodeo/mp4') #as_attachment=True)


def detect_and_annotate(video_path, output_path, model_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    model = YOLO(model_path)
    threshold = 0.2

    sound = "./Siren Sound.mp3"

    while ret:
        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                playsound(sound)

        out.write(frame)
        ret, frame = cap.read()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(debug=True)
