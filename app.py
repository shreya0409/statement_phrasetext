import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

 

app = Flask(__name__)
model = pickle.load(open('dict_model.pkl', 'rb'))

Encoding = pickle.load(open('final_encoding.pkl','rb'))
output_encoding=pickle.load(open('final_encoding_output.pkl','rb'))
l1 = list(Encoding.values())

 

@app.route('/')
def home():
    return render_template('index.html')

 

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    int_features.pop(0)
    int_features[2]=int_features[1]+'_'+int_features[2]
    int_features.pop(1)
    print(int_features)
    final_features = list(int_features)
    print(final_features)
    final_features2=[]
    for i in range(len(final_features)):
        try:    
            final_features2.append(l1[i][final_features[i]])
            print(l1[i][final_features[i]])
        except:
            return render_template('index.html', prediction_text='INCORRECT INPUT ')
    
    #    except:
     #       return render_template('index.html', prediction_text='INCORRECT INPUT ')
    

    output=[]
    for x in output_encoding:
        print(x)
        prediction = model[x].predict(np.array(final_features2).reshape(1, -1))

 

        pred = prediction[0]
        output.append(output_encoding[x][pred])

 

    return render_template('index.html', prediction_text='phrase_text are\t {}'.format(output))

 


if __name__ == "__main__":
    app.run(debug=True)