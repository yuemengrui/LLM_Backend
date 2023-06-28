import os

workers = 1
bind = "0.0.0.0:6666"
timeout = 300
daemon = True

pidfile = "./gunicorn.pid"

log_dir = './logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

access_log_format = '%(h)s "%(m)s %(U)s %(q)s" %(s)s "%(M)sms"'

logconfig_dict = {
    'version': 1,
    'disable_existing_loggers': False,

    "root": {
        "level": "INFO",
        "handlers": ["console"]
    },
    'loggers': {
        "gunicorn.error": {
            "level": "INFO",
            "handlers": ["error_file"],
            # 是否将日志打印到控制台（console），若为True（或1），将打印在supervisor日志监控文件logfile上，对于测试非常好用；
            "propagate": 0,
            "qualname": "gunicorn_error"
        },

        "gunicorn.access": {
            "level": "INFO",
            "handlers": ["access_file"],
            "propagate": 0,
            "qualname": "access"
        }
    },
    'handlers': {
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "encoding": "utf-8",
            "maxBytes": 1024 * 1024 * 100,
            "backupCount": 1,
            "formatter": "generic",
            "filename": log_dir + "/error.log"
        },
        # "access_file": {
        #     "class": "logging.handlers.RotatingFileHandler",
        #     "maxBytes": 1024 * 1024 * 10,
        #     "backupCount": 3,
        #     "formatter": "generic",
        #     "filename": log_dir + "/access.log",
        # },
        "access_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "encoding": "utf-8",
            "when": "MIDNIGHT",
            "backupCount": 30,
            "formatter": "generic",
            "filename": log_dir + "/access.log",
        },
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'generic',
        },

    },
    'formatters': {
        "generic": {
            "format": "%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)s] %(message)s",
            "datefmt": "[%Y-%m-%d %H:%M:%S]",
            "class": "logging.Formatter"
        }
    }
}
