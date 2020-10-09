
import numpy as np
import io
import tensorflow as tf
import pandas as pd
from keras.models import load_model
from flask import request
from flask import jsonify
from flask import Flask
import json
app = Flask(__name__)
graph = tf.get_default_graph()

def get_model():
    global model
    global graph
    model = load_model('model_without_stop.h5')
    model._make_predict_function()
    graph = tf.get_default_graph()
        
    #(... do inference here ...)
    print(" * Model loaded!")
def preprocess_model(data):
    #data['age'] = pd.to_numeric(data['age'], errors='coerce')
# split into input (X) and output (Y) variables
    X = data.iloc[:, 0:10]
    y = data.iloc[:,10]
    return X,y

print(" * Loading Keras model...")
get_model()


@app.route("/", methods=["GET"])
def predict():
    message = request.get_json(force=True)
    print(message)
    data = pd.DataFrame(message,index=[0])
    print(data)
    data['age'] = pd.to_numeric(data['age'], errors='coerce')
    data = data.iloc[:, 0:10]
    #y = data.iloc[:,10]
    #print(message.values())
    #temp = None
    #dictlist= []
    #encoded = message['image']
    #decoded = base64.b64decode(encoded)
    #image = Image.open(io.BytesIO(decoded))
    #for key, value in message.items():
    #    temp = value
    #    dictlist.append(temp)
    #print(dictlist)
    #print(type(dictlist))
    #data = dictlist[:, 0:10]
    #print(data)

    
    #data = [message]
    #data = [np.array(dictlist)]
    #print(data)
    #print(data.shape)
    #processed_model = preprocess_model(data)
    with graph.as_default():
        prediction = model.predict(data)
    print(prediction)
    #prediction = model.predict(processed_image).tolist()
    #score = model.evaluate(X, y, verbose=0)
    #print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    response = {
        'prediction': {
            'result' : prediction[0][0]
        }
    }
    #return jsonify(response)
    return json.dumps(str(response))

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
"please wait until server has fully started"))
    
    app.run(debug=True, port=5000)
