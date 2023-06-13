from flask import Flask, render_template, request
from classifier import read_image

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_path = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html')

        image_file = request.files['image']
        if image_file.filename == '':
            return render_template('index.html')

        

        # Save the uploaded image temporarily
        image_path = 'static/temp_image.jpg'
        image_file.save(image_path)
        result = read_image(image_path)
        if request.method == 'GET':
            result = None
            image_path = None
            
    return render_template('index.html', result=result, image_path=image_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
