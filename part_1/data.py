import fire
import tqdm
import h5py  
import numpy as np
import mmh3
import re
from keras.utils.np_utils import to_categorical
from collections import Counter
from six.moves import cPickle
from multiprocessing import Pool
from misc import get_logger, Option
opt = Option('./config.json')  

y_vocab_path = './data/y_vocab.py3.cPickle'

def _build_y_vocab(path) :  

    def get_class(h, i):
        b = h['bcateid'][i]
        m = h['mcateid'][i]
        s = h['scateid'][i]
        d = h['dcateid'][i]
        return '%s>%s>%s>%s' % (b, m, s, d)

    h = h5py.File(path, 'r')['train']
    size = h['pid'].shape[0]
    y_vocab = set()
    for idx in tqdm.tqdm(range(size), mininterval=1):
        class_name = get_class(h, idx)        
        y_vocab.add(class_name)
        
    return y_vocab

def build_y_vocab() :
     
    #y_vocab_path = './data/y_vocab.py3.cPickle'
    
    pool = Pool(opt.num_workers)
    rets = pool.map_async(_build_y_vocab,[path for path in opt.train_data_list]).get()
    pool.close()
    pool.join()
    
    temp = set()
    for _rets in rets :
        temp = temp | _rets # 합집합 연산 
            
    y_vocab = {y: idx for idx, y in enumerate(temp)}
    cPickle.dump(y_vocab, open(y_vocab_path, 'wb'))
    print('test')

    return None

def _preprocessing(data) : 
 
    def get_class(h, i):
        b = h['bcateid'][i]
        m = h['mcateid'][i]
        s = h['scateid'][i]
        d = h['dcateid'][i]
        return '%s>%s>%s>%s' % (b, m, s, d)
    def generate(data_path_list, begin_offset, end_offset, div='train'): 
        offset = 0
        for data_path in data_path_list:
            h = h5py.File(data_path, 'r')[div]
            sz = h['pid'].shape[0]
            if begin_offset and offset + sz < begin_offset:
                offset += sz
                continue
            if end_offset and end_offset < offset:
                break
            for i in range(sz):
                if not (begin_offset < offset + i <=end_offset):
                    continue
                class_name = get_class(h, i)
                yield h['pid'][i], class_name, h, i
            offset += sz
    def parse_data(label,h,i) :
        y_vocab = cPickle.loads(open(y_vocab_path, 'rb').read())
        Y = to_categorical(y_vocab.get(label), len(y_vocab))
            
        re_sc = re.compile('[\!@#$%\^&\*\(\)\-=\[\]\{\}\.,/\?~\+\'"|]')  
        
        product = h['product'][i].decode('utf-8')        
        product_ = re_sc.sub(' ', product).split() 
        words=product_
        words = [w for w in words if len(w) >= opt.min_word_length and len(w) < opt.max_word_length]

        if not words:
            return [None] * 2
        hash_func = lambda x: mmh3.hash(x, seed=17) 

        x = [hash_func(w) % opt.unigram_hash_size + 1 for w in words]
        xv = Counter(x).most_common(opt.max_len)
        
        x = np.zeros(opt.max_len, dtype=np.float32)
        v = np.zeros(opt.max_len, dtype=np.int32)
        for i in range(len(xv)):
            x[i] = xv[i][0]
            v[i] = xv[i][1]
        X = (x,v)
        
        return Y,X
    
    data_path_list, div, out_path, begin_offset, end_offset = data
    print(data_path_list, div, out_path, begin_offset, end_offset)
    
    tmp = generate(data_path_list, begin_offset, end_offset,div) 
    rets = []
        
    for pid, label, h, i in tmp:
        y, x = parse_data(label, h, i)
        if y is None:
            continue
        rets.append((pid, y, x))

    cPickle.dump(rets, open(out_path,'wb'))
    return None

def preprocessing(data_path_list, div, chunk_size) : 
    def _split_data(data_path_list, div='train', chunk_size=2000):
        total = 0
        for data_path in data_path_list:
            h = h5py.File(data_path, 'r')
            sz = h[div]['pid'].shape[0]
            total += sz
        chunks = [(i, min(i + chunk_size, total)) for i in range(0, total, chunk_size)]
        return chunks
 
    chunk_offsets = _split_data(data_path_list, div, chunk_size)
    num_chunks = len(chunk_offsets)
    tmp_chunk_tpl = 'tmp/base.chunk.%s'   
    
    pool = Pool(opt.num_workers)
    rets = pool.map_async(_preprocessing,[(data_path_list, div ,tmp_chunk_tpl%idx ,begin, end) for idx, (begin,end) in enumerate(chunk_offsets)]).get()
    pool.close()
    pool.join()


    return num_chunks


def make_db(data_name, output_dir='data/train', train_ratio=0.8) :
    if data_name == 'train':
        div = 'train'
        data_path_list = opt.train_data_list
        chunk_size = opt.chunk_size
    else : 
        return 
    
    preprocessing(data_path_list,div,chunk_size)


    return None
if __name__ == '__main__':
    
    fire.Fire({'build_y_vocab': build_y_vocab,
                'make_db': make_db})

