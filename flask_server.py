
import os
import re

import traceback

from flask import Flask, jsonify, request
from flask_cors import CORS

from importlib import import_module

from allowed.gpc.Gpc import Gpc

with Gpc() as obj:
    pass  # do this to init the model and tokenizer - yep!

flask_server_config = {

    "host": "0.0.0.0",
    "port": int(os.environ.get("PORT", 4891)),
    "debug": False,
    "threaded": True
}

app = Flask ( __name__ )
CORS(app)


@app.route('/read', methods=['GET', 'POST'])
def read():

    response = None
    result = None
    status_code = 200
    params = request.get_json() if request is not None else None
    if not params: params = request.form.to_dict()
    if not params:
        params = request.args
        if params: params=params.to_dict()
    try:
        if  params is not None :
            result = _dynamic_module(params)
        else :
            result = {"error": "no command"}
            status_code = 404
    except Exception as ex:
        result = {"error": str(ex)}
        print (traceback.format_exc())
        status_code = 404
    finally:
        response = jsonify(result)
        response.status_code = status_code
    return response


r_fn = re.compile('\.([^\.]+$)')


def _dynamic_module(params=None):

    """

    params: {
    "module_fn":"mod.mod1.mod2.fn",
    "param1":"hendrix",
    "param2":"puttin"
    }

    """
    if params is None or 'module_fn' not in params: return None
    remove_props = ['module_fn']
    module_fn = 'allowed.' + params.get('module_fn') #security
    fn = r_fn.search(module_fn).group()
    module = re.sub(re.escape(fn)+r'$', '', module_fn)
    fn = fn.replace('.','')

    # clear the unneeded stuff
    for prop in remove_props:
        if prop in params:
            del params[prop]

    module = import_module(module)
    method_to_call = getattr(module, fn)
    return method_to_call(**params) if method_to_call is not None else None


if __name__ == '__main__':
    app.run ( host=flask_server_config['host'], port=flask_server_config['port'], debug=flask_server_config['debug'],
            threaded=flask_server_config['threaded'])
