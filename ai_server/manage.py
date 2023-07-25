# *_*coding:utf-8 *_*
from info import app

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        threaded=False,
        processes=1
    )
