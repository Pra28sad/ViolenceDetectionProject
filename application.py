from flask import Flask, render_template, request, send_file, redirect, url_for
import os
from projectModel import image_pred, global_url, video_pred, live_video

application = Flask(__name__)
app = application

# Configure upload folder
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle Image Upload
        if "image" in request.files and request.files["image"].filename:
            file = request.files["image"]
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            image_pred(filepath)
            # image_pred("path/to/image.jpg")  # Process image
            return send_file("static/output.png", mimetype="image/png")
            # return render_template("index.html", message="Image processed successfully!", image_path=filepath)

        # Handle Video Upload
        elif "video" in request.files and request.files["video"].filename:
            file = request.files["video"]
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            video_pred(filepath)
            return render_template("index.html", message="Video processed successfully!", video_path=filepath)

        # Handle URL Input
        elif "url" in request.form and request.form["url"]:
            url = request.form["url"]
            global_url(url)
            return send_file("static/output.png", mimetype="image/png")
            # return render_template("index.html", message="URL processed successfully!", image_url=url)

        # Handle Live Video
        elif "live" in request.form:
            live_video()
            return render_template("index.html", message="Live detection started!")

    return render_template("index.html")




if __name__ == "__main__":
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

