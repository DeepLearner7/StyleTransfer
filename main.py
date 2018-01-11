import os
import transform, numpy as np
import tensorflow as tf
from nocache import nocache
from utils import save_img, get_img
import numpy
import jsonify
from flask import Flask, render_template, request
from flask.ext.uploads import UploadSet, configure_uploads, IMAGES


app = Flask(__name__, static_url_path='/static')

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = 'static/input/'
configure_uploads(app, photos)





def forward_prop(data_in, paths_out, checkpoint_dir, device_t='/cpu:0', batch_size=1):
    assert len(paths_out) > 0
    is_paths = type(data_in) == str
    
    img_shape = get_img(data_in).shape

    
    #print("Batch size: ", batch_size)

    g = tf.Graph()

    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True
    with g.as_default(), g.device(device_t),tf.Session(config=soft_config) as sess:
        batch_shape = (batch_size,) + img_shape
        #print("Batch_shape: ", batch_shape)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape,
                                         name='img_placeholder')

        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()
        
        #Restore checkpoint in session
        saver.restore(sess, checkpoint_dir)
        
        
        curr_batch_out = paths_out
        
        if is_paths:
            curr_batch_in = data_in
            print("curr_batch_in: ", curr_batch_in) 
            print("curr_batch_out: ", curr_batch_out)   
            X = np.zeros(batch_shape, dtype=np.float32)
            
            img = get_img(curr_batch_in)
            assert img.shape == img_shape, 'Images have different dimensions. ' + 'Resize images'
            X[0] = img
            #print("Shape: ", X.shape)  #(1,960,960,3)
        

        _preds = sess.run(preds, feed_dict={img_placeholder:X})
        
        save_img(curr_batch_out, _preds[0])
        sess.close()
    print("Done!!")
    return curr_batch_out

def begin(input_dir, style_choice):
    checkpoint_dir = 'checkpoint/'+style_choice
    
    in_path = input_dir
    out_path = 'static/output/'+in_path.split('/')[-1]
    
    curr_batch_out = forward_prop(in_path, out_path, checkpoint_dir, batch_size=1, device_t='/cpu:0')
    return curr_batch_out


@app.route('/')
@nocache
def dir1():
    return render_template("index.html")


@app.route('/uploadajax', methods=['GET', 'POST'])
@nocache
def upload():
    print(request.form.get("style"))
    if request.method == 'POST' and 'file' in request.files:
        filename = photos.save(request.files['file'])
        
        style_input = request.form.get("style")
        style_input = style_input.split('.')
        style_choice = style_input[0] + '.ckpt'
        
        input_dir = os.path.join(os.path.dirname(__file__), 'static', 'input', filename)
        print("IP:", request.remote_addr)
        generated_img_name = begin(input_dir, style_choice)
        return generated_img_name
    else:
        print('File not recieved')

    return "Failed"


if __name__ == '__main__':
    app.run(debug=True)