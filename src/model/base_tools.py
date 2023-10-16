import sys,os,time,shutil
import numpy as np
import tensorflow.compat.v1 as tf
from datetime import datetime, timedelta
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from data_loader import DataLoader
from utils import _get_hash_feat

class BaseTools(BaseEstimator, TransformerMixin):
    def __init__(self, logger,
                 **wargs):
        self.params = wargs
        self.logger = logger

    def data_split_func(self, x):
        str_lst = tf.string_split([x],'\t').values
        session_id = str_lst[0]
        itemid = str_lst[1]
        label = tf.string_to_number(tf.string_split([str_lst[2]], ',').values, out_type=tf.float64)
        user_hash_feat = _get_hash_feat(str_lst[3])
        item_hash_feat = _get_hash_feat(str_lst[4])
        cross_hash_feat = _get_hash_feat(str_lst[5])
        return session_id, itemid, label, user_hash_feat, item_hash_feat, cross_hash_feat

    def test(self):
        self.test_impl()

    def SinglePredict(self, local_predict_f):
        self.logger.info(self.params)
        tf.reset_default_graph()
        iterator = DataLoader(self.params['test_dir'], "local", self.logger).dataset_get(self.params['batch_size'], self.params['parallel'], self.data_split_func).make_initializable_iterator()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(iterator.initializer)
        self._init(iterator, False)
        with sess.as_default():
            tf.train.Saver().restore(sess, os.path.join(self.params['model_dir']+'/checkpoint', 'train_model'))
            f = open(local_predict_f, 'w')
            while True:
                try:
                    result , sid_v, item_v, label_v = sess.run([self.preds, self.session_id, self.itemid, self.label])
                    result = result.tolist()
                    sid_v = sid_v.tolist()
                    item_v = item_v.tolist()
                    label_v = label_v.tolist()
                    for i in range(0,len(result)):
                        f.write(('{}\t'*3+'{}\n').format(sid_v[i].decode(), item_v[i].decode(), \
                            '\t'.join(map(str, label_v[i])), '\t'.join(map(str, result[i]))))
                except:
                    return

    def Predict(self, local_predict_f):
        logger = self.logger
        params = self.params
        test_predict_start_time = time.time()
        self.SinglePredict(local_predict_f)
        self.logger.info("test_predict_cost:{}.".format(time.time()-test_predict_start_time))

    def test_impl(self):
        abs_model_dir = os.path.abspath(self.params['model_dir'])
        test_predict_f = abs_model_dir+'/predict_res.txt'
        metric_res_f = abs_model_dir+'/metric_res.txt'
        rm_test_f = """
            ls {dst}
        """.format(dst=test_predict_f)
        self.Predict(test_predict_f)
        self.calc_auc(test_predict_f, metric_res_f, 'w')
        self.calc_gauc(test_predict_f, metric_res_f, 'a')
        self.calc_recall(test_predict_f, metric_res_f, 'a')
        os.system(rm_test_f)
        self.logger.info('test succeed!')

    def train(self):
        self.train_impl()

    def train_impl(self):
        f_loss = open(self.params['model_dir'] + '/loss.txt', 'w')
        self.logger.info('*' * 20 + ". start train ......")
        self.SingleTrain(f_loss)
        self.logger.info('save model finished!')

    def SingleTrain(self, f_loss):
        tf.reset_default_graph()
        iterator = DataLoader(self.params['train_dir'], "local", self.logger, self.params['begin_day'], self.params['end_day']).dataset_get(self.params['batch_size'],
                                                                                            self.params['parallel'],
                                                                                            self.data_split_func).make_initializable_iterator()
        self._init(iterator, True)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        checkpoint_path = self.params['model_dir']  + '/checkpoint'
        if os.path.exists(checkpoint_path):
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(checkpoint_path, 'train_model'))
        else:
            sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
        loss_l = list()
        while True:
            try:
                _, loss_tuple = sess.run([self.optimizer, self.loss_tuple])
                loss_l.append(loss_tuple)
            except tf.errors.OutOfRangeError as e:
                f_loss.write('single train finished. train step:{}\n'.format(len(loss_l)))
                f_loss.flush()
                break
        f_loss.write('{}\n'.format('\n'.join(map(str, loss_l))))
        f_loss.flush()
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
        os.mkdir(checkpoint_path)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(checkpoint_path, 'train_model'))
        self.logger.info('save ckpt finished!')
        sess.close()

    def calc_gauc(self, predict_f, gauc_f, mod='a'):
        f_out = open(gauc_f, mod)
        def IsValid(l):
            for i in range(len(l)-1):
                if l[i] != l[i+1]:
                    return True
            return False

        def CalcGaucSingle(d):
            gauc_sum = 0.
            len_sum = 0
            for label_l, score_l in d.values():
                if IsValid(label_l):
                    gauc = roc_auc_score(label_l, score_l)
                    gauc_sum += len(label_l) * gauc
                    len_sum += len(label_l)
            return gauc_sum,len_sum

        sid_d_0 = dict()
        sid_d_1 = dict()
        for line in open(predict_f, 'r'):
            try:
                sid, itemid, label_0, label_1, label_2, score_0, score_1 = line.strip().split('\t')[0:7]
            except:
                continue
            if float(label_0) == 0.0 and float(label_1) == 0.0:
                continue
            sid_d_0.setdefault(sid, [list(), list()])
            sid_d_0[sid][0].append(int(float(label_0)))
            sid_d_0[sid][1].append(float(score_0))

            sid_d_1.setdefault(sid, [list(), list()])
            sid_d_1[sid][0].append(int(float(label_0)))
            sid_d_1[sid][1].append(float(score_1))

        gauc_sum_0, len_sum_0 = CalcGaucSingle(sid_d_0)
        gauc_sum_1, len_sum_1 = CalcGaucSingle(sid_d_1)
        f_out.write('p1-gauc\t{:.4f}\n'.format(gauc_sum_0/len_sum_0))
        if not ('forbid_p2_metric' in self.params and self.params['forbid_p2_metric']):
            f_out.write('p2-gauc\t{:.4f}\n'.format(gauc_sum_1/len_sum_1))

    def calc_auc(self, predict_f, gauc_f, mod='a'):
        f_out = open(gauc_f, mod)
        label_l = list()
        score_l_0 = list()
        score_l_1 = list()
        for line in open(predict_f, 'r'):
            try:
                sid, itemid, label_0, label_1, label_2, score_0, score_1 = line.strip().split('\t')[0:7]
            except:
                continue
            if float(label_0) == 0.0 and float(label_1) == 0.0:
                continue
            label_l.append(int(float(label_0)))
            score_l_0.append(float(score_0))
            score_l_1.append(float(score_1))
        leng = len(label_l)
        gauc_0 = roc_auc_score(label_l, score_l_0)
        gauc_1 = roc_auc_score(label_l, score_l_1)
        f_out.write('p1-auc\t{:.4f}\n'.format(gauc_0))
        if not ('forbid_p2_metric' in self.params and self.params['forbid_p2_metric']):
            f_out.write('p2-auc\t{:.4f}\n'.format(gauc_1))

    def calc_recall(self, predict_f, gauc_f, mod='a'):
        f_out = open(gauc_f, mod)
        d_0 = dict()
        d_1 = dict()
        def calc_recall_single(d):
            g_imp_cnt = 0
            g_clk_cnt = 0
            recall_d = dict()
            recall_d[1] = [0., 0.]
            recall_d[10] = [0., 0.]
            recall_d[50] = [0., 0.]
            for (k,v) in d.items():
                imp_s = v[0]
                if len(imp_s) > 0:
                    g_imp_cnt += 1
                clk_s = v[1]
                if len(clk_s) > 0:
                    g_clk_cnt += 1
                sort_l = v[2]
                sort_l.sort(key=lambda ele:ele[1], reverse=True)
                for k in recall_d.keys():
                    topk_s = set(list(zip(*sort_l[:k]))[0])
                    if len(imp_s) > 0:
                        recall_d[k][0] += float(len(imp_s & topk_s))/len(imp_s)
                    if len(clk_s) > 0:
                        recall_d[k][1] += float(len(clk_s & topk_s))/len(clk_s)
            return recall_d, g_imp_cnt, g_clk_cnt
        for line in open(predict_f, 'r'):
            try:
                sid, itemid, label_0, label_1, label_2, score_0, score_1 = line.strip().split('\t')[0:7]
            except:
                continue
            d_0.setdefault(sid, [set(), set(), list()])
            d_1.setdefault(sid, [set(), set(), list()])
            if float(label_0) == 1.0 or float(label_1) == 1.0:
                d_0[sid][0].add(itemid)
                d_1[sid][0].add(itemid)
            if float(label_0) == 1.0:
                d_0[sid][1].add(itemid)
                d_1[sid][1].add(itemid)
            d_0[sid][2].append((itemid, float(score_0)))
            d_1[sid][2].append((itemid, float(score_1)))

        recall_d_0, g_imp_cnt_0, g_clk_cnt_0 = calc_recall_single(d_0)
        recall_d_1, g_imp_cnt_1, g_clk_cnt_1 = calc_recall_single(d_1)

        f_out.write('{}\t{:<10}\t{:<10}\n'.format(' '*15, 'exposure', 'click'))
        for (k,v) in recall_d_0.items():
            f_out.write('p1-recall@{:<5}\t{:<10.2%}\t{:<10.2%}\n'.format(k, v[0]/g_imp_cnt_0, v[1]/g_clk_cnt_0))
        if not ('forbid_p2_metric' in self.params and self.params['forbid_p2_metric']):
            for (k,v) in recall_d_1.items():
                f_out.write('p2-recall@{:<5}\t{:<10.2%}\t{:<10.2%}\n'.format(k, v[0]/g_imp_cnt_1, v[1]/g_clk_cnt_1))
