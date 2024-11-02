from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

#fitur question answering
import google.generativeai as genai

#fitur display peta
import leafmap

#fitur klasifikasi
from classify_image import classify

app = Flask(__name__)
CORS(app)

APPS_URL = "http://localhost:3000/uploads/"
API = 'AIzaSyBmF2loRqAxx3Scjum_UatILg4s1pWeM2M' 

@app.route('/api/map')
def generate_map():
    # Buat peta menggunakan Leafmap
    m = leafmap.Map()
    m.split_map(left_layer="ESA WorldCover 2020 S2 FCC", right_layer="ESA WorldCover 2020")
    m.add_legend(title="ESA Land Cover", builtin_legend="ESA_WorldCover")

    # Simpan peta sebagai HTML
    map_file = 'map.html'
    m.to_html(map_file)

    # Sajikan file HTML untuk diakses oleh frontend
    return send_file(map_file)

import google.generativeai as genai

@app.route('/api/gemini', methods=['POST'])
def generate_response():
    data = request.get_json()
    question = data.get('question') + " Berikanlah jawabannya secara akurat, singkat, dan ringkas!"

    genai.configure(api_key=API)
    model = genai.GenerativeModel("gemini-1.5-flash")

    response = model.generate_content(question)
    return jsonify({'answer' : response.text})


@app.route('/api/klasifikasi', methods=['POST'])
def classify_photo():
    try:
        data = request.json
        image_path = data.get('image_path')
        full_path = APPS_URL + image_path
        if not image_path:
            return jsonify({'error': 'Tidak ada gambar yang disediakan', 'success': False}), 400

        #menerapkan model klasifikasi
        labels = classify(full_path)

        data = {'name': image_path, 'descriptions': labels, 'success': True}
        #data = {'name': full_path, 'success': True}
        return jsonify(data),200
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)