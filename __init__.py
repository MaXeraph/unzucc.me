import os, math
from flask import Flask, render_template, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import face_recognition
import cv2
from PIL import Image
import numpy as np

fileDir = os.path.dirname(os.path.realpath('__file__'))

UPLOAD_FOLDER = os.path.join(fileDir, 'static')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
filename = "downloadButton.png"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 300

def fuzz(file,faces,level):
    src1 = np.array(Image.open(file))

    noise = np.rint(np.random.normal(0, level, src1.shape))
    mask = np.zeros(src1.shape, dtype=int)

    for face in faces:
        mask[face[0]:face[2], face[3]:face[1]] = noise[face[0]:face[2], face[3]:face[1]]

    final = np.clip(np.absolute(np.add(mask, src1)).astype(int), 0, 255)
    im = Image.fromarray(final.astype('uint8'), 'RGB')
    im.save(file)

@app.route("/")
def home():
	return render_template("index.html", imaging=filename)


def allowed_file(fname):
    return '.' in fname and \
           fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        data = request.form["text"]
        if data == "":
            mask_level = 1
        else:
            mask_level = abs(int(data.upper()))

        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            destination = "./static/" + filename
            image = face_recognition.load_image_file("./static/"+filename)
            face_locations = face_recognition.face_locations(image)
            fuzz(destination, face_locations, mask_level)

            return render_template("index.html", imaging=filename)

    return render_template("index.html", imaging=filename)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

if __name__ == "__main__":
	app.run(debug=True)


