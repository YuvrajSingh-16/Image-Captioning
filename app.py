from flask import Flask, request, render_template
import Caption_It

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template("index.html")
    

@app.route('/', methods=['POST'])
def caption():
    if request.method == 'POST':
        f = request.files['userfile']
        path = "./static/{}".format(f.filename)
        f.save(path)
        
        img = Caption_It.encode_image(path)
        caption = Caption_It.predict_caption(img)
        print(caption)
        
        result = {
            "Image": path,
            "Caption": caption
        }
        
    return render_template("index.html", ur_result = result)


if __name__ == '__main__':
    app.run(debug=True)
    
    