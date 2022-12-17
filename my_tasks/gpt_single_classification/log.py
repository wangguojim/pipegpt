# -*- coding: UTF-8 -*-
# @Author ：Mr. QingQuan
# @Date   ：2020/8/11 11:47 上午
# -*- coding: utf-8 -*-
import os
import time
import logging
import logging.handlers
abs_path = __file__[:__file__.rfind('/')]


class Logger:

    def __init__(self, log_path=False, console=True, save_mode='file', level=logging.INFO):
        self.level = level
        self.logger = logging.getLogger()  # 创建logger
        self.logger.setLevel(level)  # log等级总开关

        self.create_handler(log_path, mode=save_mode)
        if console:
            self.create_handler(log_path, mode='console')

    def create_handler(self, log_path, mode='console'):
        if mode == 'console':
            handler = logging.StreamHandler()
        elif mode == 'file':
            handler = logging.FileHandler(self.get_logfile_name(log_path), mode='a', encoding='utf-8')
        elif mode == 'rotat':
            handler = logging.handlers.RotatingFileHandler(self.get_logfile_name(log_path), mode='a', maxBytes=1e8, encoding='utf-8')
        elif mode == 'time_rotat':
            handler = logging.handlers.TimedRotatingFileHandler(self.get_logfile_name(log_path),  when="D", interval=1,
                                                     backupCount=30, encoding='utf-8')
        else:
            raise Exception('you must secect one in ["console", "file", "rotat", "time_rotat"]!')
        handler.setLevel(self.level)  # 输出到file的log等级的开关
        handler.setFormatter(Logger.formatter())
        self.logger.addHandler(handler)  # 将logger添加到handler里

    def get_logfile_name(self, log_path):
        log_name = abs_path+ '/' + log_path + '/' + Logger.current_time(for_file_name=True) + '.log'
        file_path, file_name = os.path.split(log_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        return log_name

    @staticmethod
    def formatter():
        return logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

    @staticmethod
    def current_time(for_file_name=False):
        if not for_file_name:
            return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        else:
            return time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))


if __name__ == "__main__":
    pass

    logger = Logger(save_mode='rotat', log_path='logs/').logger
    for i in range(100):
        # time.sleep(0.1)
        logger.info('hello')
        logger.info('thanks')

    


