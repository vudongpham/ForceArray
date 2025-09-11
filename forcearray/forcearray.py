from datetime import datetime, timedelta
from itertools import product
import os
import rasterio
import numpy as np
import fastnanquantile as fnq


def toRaster(arr_in, base_raster, out_raster):
    with rasterio.open(base_raster) as src:
        band_crs = src.crs
        band_tranform = src.transform

    meta = {
        "driver": "GTiff",
        "height": arr_in.shape[1],
        "width": arr_in.shape[2],
        "count": arr_in.shape[0],  
        "dtype": np.int16,
        "crs": band_crs,  
        "transform": band_tranform,
        "nodata": -9999,  
    }

    with rasterio.open(out_raster, "w", **meta) as dst:
        for i in range(arr_in.shape[0]):
            dst.write(arr_in[i], i + 1)

class forcearray():
    def __init__(self, level2_dir, tile, start_date, end_date, sensors, best_cso=False):
        self.tile_dir = os.path.join(level2_dir, tile)
        self.start_date = start_date
        self.end_date = end_date
        self.sensors = sensors
        self.best_cso = best_cso

        # Get list boa, qai
        self.boa_files, self.qai_files = self.__get_image_list()

        # Get cso value
        self.cso_list = self.__get_cso_value()
    

    def get_data_te(self, time_step, nodata=-9999, toInt16=False, trackProgress=False):
        data_dates = np.array([x[:8] for x in self.qai_files])
        data_dates = np.array([datetime.strptime(date_str, "%Y%m%d") for date_str in data_dates], dtype=np.datetime64)
        
        qai_stack = np.array([self.__read_image(qai) for qai in self.qai_files])
        boa_stack = np.array([self.__read_image(boa) for boa in self.boa_files])
        nodata_mask = ~np.isin(qai_stack, self.cso_list)
        boa_stack[nodata_mask, :] = nodata

        del qai_stack

        target_dates = self.__generate_date_list(time_step)
        
        te = np.full(shape=(len(target_dates) - 1, boa_stack.shape[1], boa_stack.shape[2], boa_stack.shape[3]), fill_value=nodata)
        
        progress = len(target_dates)
        c = 0
        for i in range(len(target_dates) - 1):
            mask = np.logical_and(data_dates >= target_dates[i], data_dates < target_dates[i + 1])
            if not np.any(mask):
                c += 1
                continue
            x_mask = boa_stack[mask]
            x_mask = np.ma.masked_equal(x_mask, nodata)
            x_mask = np.ma.mean(x_mask, axis=0)
            x_mask = x_mask.filled(nodata)
            te[i, ...] = x_mask
            c += 1
            if trackProgress:
                print(f'\rEncoding {c}/{progress}', end='', flush=True)
        
        del boa_stack 
        if toInt16:
            te = te.astype(np.int16)
        return te
    
    def get_data_stm(self, p_list, nodata=-9999, toInt16=False):
        boa_stack = np.array([self.__read_image(boa) for boa in self.boa_files])
        qai_stack = np.array([self.__read_image(qai) for qai in self.qai_files])
        nodata_mask = ~np.isin(qai_stack, self.cso_list)
        boa_stack[nodata_mask, :] = nodata
        boa_stack = boa_stack.astype(np.float32)
        boa_stack = np.where(boa_stack == nodata, np.nan, boa_stack)
        stm = fnq.nanquantile(boa_stack, q=p_list, axis=0)
        stm = np.nan_to_num(stm, copy=False, nan=nodata)
        del boa_stack
        if toInt16:
            stm = stm.astype(np.int16)
        return stm
    
    def get_data_mean(self, nodata=-9999, toInt16=False):
        boa_stack = np.array([self.__read_image(boa) for boa in self.boa_files])
        qai_stack = np.array([self.__read_image(qai) for qai in self.qai_files])
        nodata_mask = ~np.isin(qai_stack, self.cso_list)
        boa_stack[nodata_mask, :] = nodata
        boa_stack = np.ma.masked_equal(boa_stack, nodata)
        mean_stack = np.ma.mean(boa_stack, axis=0)
        mean_stack = mean_stack.filled(nodata)
        del boa_stack
        if toInt16:
            mean_stack = mean_stack.astype(np.int16)
        return mean_stack
        


    def __generate_date_list(self, time_step):
        start = datetime.strptime(self.start_date, "%Y%m%d")
        end = datetime.strptime(self.end_date, "%Y%m%d")
        date_list = []
        d = start
        noleap_days = 0
        
        # Add start if it's not Feb 29
        if not (d.month == 2 and d.day == 29):
            date_list.append(d.strftime("%Y%m%d"))
    
        d += timedelta(days=1)
        while d <= end:
            # Skip Feb 29 completely
            if d.month == 2 and d.day == 29:
                d += timedelta(days=1)
                continue
    
            noleap_days += 1
            if noleap_days % time_step == 0:
                date_list.append(d.strftime("%Y%m%d"))
            d += timedelta(days=1)
    
        date_list = np.array(
            [datetime.strptime(date_str, "%Y%m%d") for date_str in date_list],
            dtype=np.datetime64
        )
        return date_list
    
    def __read_image(self, image_name, nodata=-9999):
        with rasterio.open(os.path.join(self.tile_dir, image_name)) as src:
            band_count = src.count
            arr = src.read()
        if band_count > 1:
            arr = np.moveaxis(arr, 0, -1)
            invalid_mask = np.logical_or(np.any(arr < 0, axis=-1), np.any(arr > 10000, axis=-1))
            arr[invalid_mask, :] = nodata
        else:
            arr = arr[0]
        return arr   
    
    def __get_image_list(self):
        start_dt = datetime.strptime(self.start_date, "%Y%m%d")
        end_dt = datetime.strptime(self.end_date, "%Y%m%d")
        boa_files = []
        for filename in os.listdir(self.tile_dir):
            if filename.endswith("BOA.tif"):
                try:
                    file_date_str = filename[:8]  # Extract YYYYMMDD
                    file_date = datetime.strptime(file_date_str, "%Y%m%d")
                    
                    if start_dt <= file_date <= end_dt:
                        boa_files.append(filename)
                except ValueError:
                    continue  # Skip files that don't match the expected pattern
        if self.sensors == 'all':
            pass
        else:
            sensors_list = self.sensors.split(',')
            boa_files = [image for image in boa_files if any(sensor in image for sensor in sensors_list)]
        boa_files.sort()
        qai_files = [image.replace('BOA', 'QAI') for image in boa_files]
        return boa_files, qai_files
    
    def __get_cso_value(self):
        filtering_default = {
            'Valid data' : ['0'],
            'Cloud state' : ['00'],
            'Cloud shadow flag' : ['0'],
            'Snow flag' : ['0'],
            'Water flag': ['0', '1'],
            'Aerosol state' : ['00', '01', '10', '11'],
            'Subzero flag' : ['0'],
            'Saturation flag' : ['0'],
            'High sun zenith flag' : ['0', '1'],
            'Illumination state' : ['00', '01', '10', '11'],
            'Slope flag' : ['0', '1'],
            'Water vapor flag' : ['0', '1'],
            'Empty' : ['0']
        }

        filtering_best = {
            'Valid data' : ['0'],
            'Cloud state' : ['00'],
            'Cloud shadow flag' : ['0'],
            'Snow flag' : ['0'],
            'Water flag': ['0', '1'],
            'Aerosol state' : ['00'],
            'Subzero flag' : ['0'],
            'Saturation flag' : ['0'],
            'High sun zenith flag' : ['0'],
            'Illumination state' : ['00'],
            'Slope flag' : ['0', '1'],
            'Water vapor flag' : ['0', '1'],
            'Empty' : ['0']
        }

        if self.best_cso:
            filtering_list = filtering_best
        else:
            filtering_list = filtering_default
        cso_value = [''.join(p) for p in product(*filtering_list.values())]
        cso_value = [x[::-1] for x in cso_value]
        cso_value = [int(x, 2) for x in cso_value]
        cso_value.sort()
        return cso_value
            
    


