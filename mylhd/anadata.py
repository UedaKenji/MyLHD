import numpy as np
import matplotlib.pyplot as plt

#@title Function definition for open data manipulation
import os
import io
import random
import string
import urllib3
import traceback
import re
import numpy as np
import pandas as pd

BASEURL="https://exp.lhd.nifs.ac.jp/opendata/LHD/webapi.fcgi?cmd=getfile&diag=%s&shotno=%d&subno=%d"

def randfilename(n, template="tmp%s.dat"):
    """
    create random name
    n : number of characters
    template : template of file name
    """
    ary = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
    str = ''.join(ary)
    return template % (str)

def retrieve(diag, shotno, subno=1):
    """
    read kaiseki data from Kaiseki Server
    """
    filename = randfilename(10)
    data = None
    try:
        cmd="igetfile -d %s -s %d -m %d -o %s" % (diag, shotno, subno, filename)
        #print(cmd)
        os.system(cmd)
        if os.path.exists(filename):
            data = KaisekiData(filename)
            os.remove(filename)
    except:
        print(traceback.format_exc())
        pass
    return data

def retrieve_opendata(diag, shotno, subno=1):
    """
    read kaiseki data from open data server.
    """
    url=BASEURL % (diag, shotno, subno)
    http=urllib3.PoolManager()
    data = None
    try:
        resp = http.request('GET', url)
        fileio=io.StringIO(resp.data.decode('utf-8'))
        data = KaisekiData(fileio=fileio)
    except:
        print(traceback.format_exc())
        pass
    
    
    return data

time_names = ['Time', 'time','Time(s)', 'time(s)', 't(s)', 'T(s)']    
freq_names = ['Frequency', 'freq', 'Freq', 'frequency']
r_names    = ['R', 'r', 'R(m)', 'r(m)', 'R(m)', 'r(m)']

    
class KaisekiData:
    """
    [Class] Kaiseki Data Class
    """

    @classmethod
    def retrieve_opendata(cls, diag:str, shotno:int, subno:int=1) -> 'KaisekiData':
        """
        read kaiseki data from open data server.
        """
        url=BASEURL % (diag, shotno, subno)
        http=urllib3.PoolManager()
        data = None
        try:
            resp = http.request('GET', url)
            fileio=io.StringIO(resp.data.decode('utf-8'))
            data = cls(fileio=fileio)
        except:
            print(traceback.format_exc())
            pass
        return data
    
    @classmethod
    def retrieve(cls, diag:str, shotno:int, subno=1 ) -> 'KaisekiData':
        """
        read kaiseki data from Kaiseki Server
        """
        filename = randfilename(10)
        data = None
        try:
            cmd="igetfile -d %s -s %d -m %d -o %s" % (diag, shotno, subno, filename)
            #print(cmd)
            os.system(cmd)

            if os.path.exists(filename):
                data = cls(filename)
                os.remove(filename)
            else:
                raise Exception("File Not Found")
        except:
            print(traceback.format_exc())
            pass
        return data


    force_sort = True 
    # data must be sorted by dim[N-1], dim[N-2],..., dim0, in this order
    # Howerver, some data does not follow this rule, for example, tsmesh, etc.
    # If this class variable is True, the parser forces to sort data as expected.
    # However it might consume memory and cpu power for large data.

    def __init__(self, filename=None, fileio=None):

        if ( filename ):
            self.load(filename)
        elif (fileio ):
            self.parse(fileio)
        else:
            self.name = None
            self.date = None
            self.shotno = None
            self.subno = None
            self.dimnames = None
            self.dimunits = None
            self.valnames = None
            self.valunits = None
            self.dimno = None
            self.valno = None
            self.dimsizes = None
            self.data = None
            self.comment = None
            self.dimname_idx = None
            self.valname_idx = None
            
    def load(self, filename:str):
        """
        load kaiseki data from a file
        filename : file name of Kaiseki Data
        """
        try:
            with open(filename) as fp:
                self.parse(fp)
        except:
            raise Exception("File Not Open: "+filename)

    def parse(self, fp):
        """
        read kaiseki data from io stream, and analalze.
        fp: io stream like object
        """
        AREA_NONE = 0
        AREA_PARAMETER = 1
        AREA_COMMENT = 2
        AREA_DATA = 3
        RE_HEADER = re.compile(r'^# *')
        RE_PARAMETER = re.compile(r'^# *\[Parameters\]', re.IGNORECASE)
        RE_COMMENT = re.compile(r'^# *\[Comments\]', re.IGNORECASE)
        RE_DATA = re.compile(r'^# *\[Data\]', re.IGNORECASE)
        h = {}
        comment = ""
        try:
            area = AREA_NONE
            l = fp.readline()
            while l:
                
                while True:
                    if RE_PARAMETER.search(l):
                        area = AREA_PARAMETER
                    elif RE_COMMENT.search(l):
                        area = AREA_COMMENT
                    elif RE_DATA.search(l):
                        area = AREA_DATA
                    else:
                        break
                    l = fp.readline()

                if ( area == AREA_COMMENT and RE_HEADER.search(l)):
                    comment += re.sub(RE_HEADER,'',l)
                l = l.strip()
                if len(l) == 0:
                    l = fp.readline()
                    continue
                if l[0] == '#':
                    f = l[1:].split('=')
                    if len(f) == 2:
                        h[f[0].strip().lower()] = f[1].strip()
                else:
                    break
                l = fp.readline()
                continue
            fp.seek(0)
            data = np.genfromtxt(fp, delimiter=',')
        except:
            raise Exception("File Read Error")

        try:
            name     = h.get('name')
            date     = h.get('date')
            shotno   = h.get('shotno')
            subno    = h.get('subshotno')
            dimno    = h.get('dimno')
            dimsizes = h.get('dimsize')
            valno    = h.get('valno')
            dimnames = h.get('dimname')
            dimunits = h.get('dimunit')
            valnames = h.get('valname')
            valunits = h.get('valunit')
            if not subno:
                subno = 1

            self.name = name.strip("' ")
            self.date = date.strip("' ")
            self.shotno = int(shotno)
            self.subno = int(subno)
            self.dimno = int(dimno)
            self.valno = int(valno)
            self.dimnames = [c.strip("' ") for c in dimnames.split(',')] 
            self.dimunits = [c.strip("' ") for c in dimunits.split(',')] 
            self.valnames = [c.strip("' ") for c in valnames.split(',')] 
            self.valunits = [c.strip("' ") for c in valunits.split(',')] 
            self.dimsizes = [int(c.strip()) for c in dimsizes.split(',')]
            self.comment  = comment

            if ( KaisekiData.force_sort ):
                key=[]
                for i in range(self.dimno):
                    key.append(data[:,self.dimno-i-1])
                    idx = np.lexsort(key)
                data = data[idx]

            self.data = data.reshape(tuple(self.dimsizes)+(-1,))

            self._init()



        except:
            raise Exception("Illegal Format")
    
    def _init(self):
    
        self.dimname_idx = {}
        for idx, n in enumerate(self.dimnames):
            self.dimname_idx[n] = idx
        self.valname_idx = {}
        for idx, n in enumerate(self.valnames):
            self.valname_idx[n] = idx

        self.time_key = None
        for n in self.dimnames:
            if n in time_names:
                self.time_key = n
                break
        self.freq_key = None
        for n in self.valnames:
            if n in freq_names:
                self.freq_key = n
                break
        self.r_key = None
        for n in self.dimnames:
            if n in r_names:
                self.r_key = n
                break


        


    def get_dim_data(self, id:int|str) -> np.ndarray:
        """
        get dimention data as 1-D array.
        id : index of dimension, or name of dimension
        """
        if isinstance(id, int):
            n = id
        else:
            n = self.dimname_idx.get(id)
            
        if ( not (n is not None and n >= 0 and n < self.dimno )):
            raise Exception("Index Range Error id=" +id)

        idx = [0] * (self.dimno+1)
        idx[n] = slice(None)
        idx[self.dimno] = n
        return self.data[tuple(idx)].flatten()
        
    def get_val_data(self, id:int|str) -> np.ndarray:
        """
        get variable data as multi dimensional array ( dim0 x dim1 x dim2 ...)
        id : index of variable, or name of variable
        """
        if isinstance(id, int):
            n = id
        else:
            n = self.valname_idx.get(id)
            
        if ( not (n is not None and n >= 0 and n < self.valno )):
            raise Exception("Index Range Error id=" +id)

        idx = [slice(None)] * (self.dimno+1)
        idx[self.dimno] = self.dimno+n
        return self.data[tuple(idx)]

    def get_dim_unit(self, id:int|str) -> str:
        """
        get unit of dimention data
        
        id : index of dimension, or name of dimension
        """
        if isinstance(id, int):
            n = id
        else:
            n = self.dimname_idx.get(id)
            
        if ( not (n is not None and n >= 0 and n < self.dimno )):
            raise Exception("Index Range Error id=" +id)
        
        return self.dimunits[n]
        
    def get_val_unit(self, id:int) -> str:
        """
        get unit of variable 
        
        id : index of variable, or name of dimension
        """
        if isinstance(id, int):
            n = id
        else:
            n = self.valname_idx.get(id)
            
        if ( not (n is not None and n >= 0 and n < self.valno )):
            raise Exception("Index Range Error id=" +id)
        
        return self.valunits[n]
    
        
    def get_dim_size(self, id:int|str) -> int:
        """
        get size of dimention data
        
        id : index of dimension, or name of dimension
        """
        if isinstance(id, int):
            n = id
        else:
            n = self.dimname_idx.get(id)
            
        if ( not (n is not None and n >= 0 and n < self.dimno )):
            raise Exception("Index Range Error id=" +id)
        
        return self.dimsizes[n]
    
    val_unit = get_val_unit
    dim_unit = get_dim_unit
    val_data = get_val_data
    dim_data = get_dim_data
    
    @property
    def time(self) -> np.ndarray:
        """
        get time data as 1-D array.
        """
        #self._time が存在するか？
        if hasattr(self, '_time'):
            return self._time
        elif self.time_key is None:
            raise Exception("Time Key Not Found")
        else:
            return self.get_dim_data(self.time_key)
    
    def convert_key(self, key):
        """
        convert key to index or name
        key : tuple of (dim0, dim1, dim2, ..., val)
        """
        if key in time_names: 
            return self.time_key
        
        elif key in freq_names: 
            return self.freq_key
        
        elif key in r_names: 
            return self.r_key

        else:
            return key


    
    def __str__(self):
        s = "Name:{}\nShotNo:{}\nSubNo:{}\nDims={}\nVals={}\nData Size:{}".format(
            self.name, self.shotno, self.subno, self.dimnames, self.valnames, self.data.shape)
        return s
    
    def __getitem__(self, key):
        """
        get data by key
        key : tuple of (dim0, dim1, dim2, ..., val)
        """

        if isinstance(key, int):
            return self.get_val_data(key)
        elif isinstance(key, str):
            key = self.convert_key(key)
            if key in self.dimname_idx:
                return self.get_dim_data(key)
            elif key in self.valname_idx:
                return self.get_val_data(key)
            else:
                raise Exception("Illegal Key Type")
        else:
            raise Exception("Illegal Key Type")
    
    def show(self):
        
        """
        print information of Kaiseki Data
        """
        print(f'#name: {self.name}  #shotno: {self.shotno}  #subno: {self.subno}  #date: {self.date}')
        print(f'Data Shape: {self.data.shape}')
        #printするとき、一行目にdimnamesと２行目にdimunitsのリスト表示するが、その際、それぞれカンマが同じ位置になるように調整する。
        # 各セルを文字列としてクォート付きに変換
        rows = [self.dimnames, self.dimunits]
        quoted_rows = [[f"'{cell}'" for cell in row] for row in rows]

        # 列ごとの最大幅を計算（クォート込み）
        col_widths = [max(len(row[i]) for row in quoted_rows) for i in range(len(quoted_rows[0]))]

        # 整形して出力
        formatted = ", ".join(f"{cell:<{col_widths[i]}}" for i, cell in enumerate(quoted_rows[0]))
        print( f'dimnames:  {formatted}')
        formatted = ", ".join(f"{cell:<{col_widths[i]}}" for i, cell in enumerate(quoted_rows[1]))
        print( f'dimunits:  {formatted}')

        rows = [self.valnames, self.valunits]
        quoted_rows = [[f"'{cell}'" for cell in row] for row in rows]
        col_widths = [max(len(row[i]) for row in quoted_rows) for i in range(len(quoted_rows[0]))]
        formatted = ", ".join(f"{cell:<{col_widths[i]}}" for i, cell in enumerate(quoted_rows[0]))
        print( f'valnames:  {formatted}')   
        formatted = ", ".join(f"{cell:<{col_widths[i]}}" for i, cell in enumerate(quoted_rows[1]))
        print( f'valunits:  {formatted}')


    

