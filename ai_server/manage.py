# *_*coding:utf-8 *_*
import os
os.environ['NUMEXPR_MAX_THREADS'] = r'1'
from info import app

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        threaded=False,
        processes=1
    )
