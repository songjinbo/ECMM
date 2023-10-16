# coding=utf-8
import sys, os, logging
import tensorflow as tf
import tensorflow.compat.v1 as tf1
tf.compat.v1.disable_eager_execution()
os.chdir(os.path.join(os.getcwd() + '/../src/'))
sys.path.append('./src/model/')
sys.path.append('./src/read_data/')
sys.path.append('./configure/'+sys.argv[1])
sys.path.insert(0, './src')
from conf import params
from model.ecm import ECM
from model.cold import COLD
from model.esmm import ESMM
from model.vector_product import VECTOR_PRODUCT
from model.ecmm import ECMM
from model.fscd import FSCD

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s')
logger = logging.getLogger(__name__)

def run_main():
    if os.path.isdir(params['model_dir']):
        os.system('rm -rf ' + params['model_dir'])
    os.system('mkdir -p ' + params['model_dir'])
    logger.info(params)
    _model = globals()[params['model_name']]
    _model(logger, **params).train()
    _model(logger, **params).test()

def main():
    run_main()

if __name__ == '__main__':
    run_main()
