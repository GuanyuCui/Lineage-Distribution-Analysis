import os
import os.path as osp
import re
import time
import requests
import json
import numpy as np
import pandas as pd
from datetime import date, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from tqdm import tqdm

import readline

# 计时器装饰器
def timer(func):
	def wrapper(*args, **kwargs):
		s = time.time()
		result = func(*args, **kwargs)
		e = time.time()
		print(f'Total time: {round(e - s, 3)} seconds.\n')
		return result
	return wrapper

# 查询子任务, 用于并行查询
def query_task(idx : int, region : str, input_list : list, output_dict : dict, input_lock, output_lock) -> pd.DataFrame:
	# API 链接
	url = 'https://ngdc.cncb.ac.cn/ncov/api/es/genome/query'
	while True:
		input_lock.acquire()
		if len(input_list) == 0:
			print('Process {}: Exiting...'.format(idx))
			input_lock.release()
			break
		# 从队列获取数据
		query_params = input_list.pop()
		try:
			start_date = query_params['minCollectDate']
			end_date = query_params['maxCollectDate']
		except:
			start_date = query_params['minSubDate']
			end_date = query_params['maxSubDate']
		print('Process {}: Querying [{}, {}]'.format(idx, start_date, end_date))
		input_lock.release()
		# 尝试查询
		while True:
			try:
				response = requests.get(url, params = query_params)
				break
			except KeyboardInterrupt:
				raise
			except:
				print('Process {}: Query error, retrying...'.format(idx))
		content = json.loads(response.content)
		
		# 处理数据
		if len(content['result']['data']) > 0:
			# 从 json 字典构造 Dataframe
			df = pd.DataFrame.from_dict(content['result']['data'])
			# 只要六列
			df = df.loc[:, ['accession', 'country', 'province', 'lineage', 'collectDate', 'submitDate']]
			# 删除部分数据
			# 删除省份、谱系的 NA 值
			df = df.dropna(axis = 0, how = 'any', subset = ['province', 'lineage', 'collectDate', 'submitDate'])
			df = df.drop(df[df['lineage'] == 'NA'].index)
			df = df.drop(df[df['province'] == 'NA'].index)
			df = df.drop(df[df['province'] == ''].index)
			df = df.drop(df[df['province'] == ' '].index)
			if region == 'ChinaMainland':
				# 排除港澳台地区数据
				df = df.drop(df[df['province'] == 'Hong Kong'].index)
				df = df.drop(df[df['province'] == 'Macau'].index)
				df = df.drop(df[df['province'] == 'Macao'].index)
				df = df.drop(df[df['province'] == 'Taiwan'].index)
			elif region in ['Hong Kong', 'Macau', 'Taiwan']:
				df = df[df['province'] == region]
			# 删除未分类的数据
			df = df.drop(df[df['lineage'] == 'Unassigned'].index)
			df = df.drop(df[df['lineage'] == 'unclassifiable'].index)
			# 删除日期不完整的数据
			df = df.drop(df[df['collectDate'].str.len() < 10].index)
			df = df.drop(df[df['submitDate'].str.len() < 10].index)

			# 删除奇怪换行
			df = df.replace(r'\r+|\n+|\t+','', regex = True)
			# 排序并添加
			df = df.sort_values(by = ['collectDate', 'lineage'], ascending = True)
		else: 
			df = pd.DataFrame()

		output_lock.acquire()
		output_dict[start_date] = df
		print('Process {}: Query succeeded.'.format(idx))
		output_lock.release()

@timer
def get_data(region : str, start_date : str = '2019-12-20', end_date : str = '', 
			query_interval : int = 7, retry_times : int = 5, 
			num_workers : int = 10,
			cached : bool = True, is_collect_date : bool = True) -> pd.DataFrame:
	# 确定地区名
	region = '' if region == 'World' else region
	
	# 确定结束日期, 空字符串为默认今天
	end_date = date.today() if (end_date == '' or end_date is None) else date.fromisoformat(end_date)
	
	# 缓存文件名
	cache_file = 'data ' + region + ' ' + str(start_date) + ' ' + str(end_date) + '.csv'

	# 如果有缓存文件, 直接读取
	if osp.exists(cache_file):
		data = pd.read_csv(cache_file)
		print('Loaded from cache file.\n')
		print('Done. #Total Sequences: {}\n'.format(len(data)))
		return data
	
	mgr = mp.Manager()
	# 
	input_list = mgr.list()
	output_dict = mgr.dict()

	# 锁
	input_lock = mgr.Lock()
	output_lock = mgr.Lock()

	left = date.fromisoformat(start_date)
	right = left + timedelta(days = query_interval - 1)

	sorted_keys = []

	while left <= end_date:
		if right > end_date:
			right = end_date

		sorted_keys.append(str(left))

		if is_collect_date:
			params = {'country': 'China' if region in ['China', 'ChinaMainland', 'Hong Kong', 'Macau', 'Taiwan'] else region, 'host': 'Homo sapiens', 'minCollectDate': str(left), 'maxCollectDate': str(right), 'complete': 'Complete','start': 0, 'length': 100000}
		else:
			params = {'country': 'China' if region in ['China', 'ChinaMainland', 'Hong Kong', 'Macau', 'Taiwan'] else region, 'host': 'Homo sapiens', 'minSubDate': str(left), 'maxSubDate': str(right), 'complete': 'Complete','start': 0, 'length': 100000}
		
		# 添加查询字典
		input_list.append(params)

		# 下一个区间
		left = right + timedelta(days = 1)
		right = left + timedelta(days = query_interval - 1)
		
	# 并行
	processes = []

	for _ in range(num_workers):
		p = mp.Process(target = query_task, args = (_, region, input_list, output_dict, input_lock, output_lock, ))
		processes.append(p)
		p.start()

	# 等待所有进程完成
	for p in processes:
		p.join()

	# 存放所有结果的 DataFrame
	data = pd.DataFrame()

	print('Concatenating...')
	for key in tqdm(sorted_keys):
		try:
			data = pd.concat([data, output_dict[key]])
		except:
			print('Warning: key {} does not found.'.format(key))
	print('Done.')
	
	data = data.reset_index(drop = True)
	if cached:
		data.to_csv(cache_file, index = False)

	print('Done. #Total Sequences: {}\n'.format(len(data)))
	
	return data

def update_data(region, cache_file = '', query_interval = 7, retry_times = 5, cached = True):
	if cache_file == '' or cache_file is None:
		filenames = os.listdir()
		r = re.compile('data {}'.format(region))
		cache_files = list(filter(r.match, filenames))

		if len(cache_files) == 0:
			return get_data(region = region, query_interval = query_interval, retry_times = retry_times)
		else:
			cache_file = cache_files[0]

	# 读取旧文件
	data = pd.read_csv(cache_file)

	start_date = date.fromisoformat(cache_file.split(' ')[2])
	end_date = date.today()

	# 新缓存文件
	new_cache_file = 'data ' + region + ' ' + str(start_date) + ' ' + str(end_date) + '.csv'

	print('Querying recently collected sequences...')
	new_data_collect = get_data(region = region, start_date = data['collectDate'].max(), end_date = str(end_date), query_interval = query_interval, retry_times = retry_times, cached = False, is_collect_date = True)
	print('Querying recently submitted sequences...')
	new_data_submit = get_data(region = region, start_date = data['submitDate'].max(), end_date = str(end_date), query_interval = query_interval, retry_times = retry_times, cached = False, is_collect_date = False)
	
	# 删除奇怪的换行
	new_data_collect = new_data_collect.replace(r'\r+|\n+|\t+','', regex = True)
	new_data_submit = new_data_submit.replace(r'\r+|\n+|\t+','', regex = True)

	data = pd.concat([data, new_data_collect, new_data_submit])
	data = data.sort_values(by = ['collectDate', 'lineage'], ascending = True)
	# 去重
	data = data.drop_duplicates(subset = ['accession'])
	data = data.reset_index(drop = True)
	if cached:
		data.to_csv(new_cache_file, index = False)
		os.remove(cache_file)

	print('Done. After duplicating, #Total Sequences: {}\n'.format(len(data)))

	return data

@timer		
def process_lineage(data, fine_grained = True, start_date = '2019-12-24', end_date = '', interval = 28, step = 14, query_start_date = '2019-12-24'):
	# 先做一遍处理
	print('Preprocessing lineages...')
	dataframe = data.copy()
	lineages = dataframe['lineage']
	# 替换别名
	alias_dict = {'AD': 'B.1.1.315', 'AE': 'B.1.1.306', 'AU': 'B.1.466.2', 'AY': 'B.1.617.2', 'AZ': 'B.1.1.318', 'BE': 'BA.5.3.1', 'BM': 'BA.2.75.3', 'BR': 'BA.2.75.4', 'C': 'B.1.1.1', 'CK': 'BA.5.2.24', 'CL': 'BA.5.1.29', 'CM': 'BA.2.3.20', 'CQ': 'BA.5.3.1.4.1.1', 'CR': 'BA.5.2.18', 'D': 'B.1.1.25', 'DB': 'BA.5.2.25', 'DN': 'BQ.1.1.5', 'DS': 'BN.1.3.1', 'DT': 'BQ.1.1.32', 'DU': 'BQ.1.1.2', 'DV': 'CH.1.1.1', 'DZ': 'BA.5.2.49', 'EA': 'BQ.1.1.52', 'ED': 'BQ.1.1.18', 'EF': 'BQ.1.1.13', 'EG': 'XBB.1.9.2', 'EJ' : 'BN.1.3.8', 'EK': 'XBB.1.5.13', 'EL': 'XBB.1.5.14', 'EM': 'XBB.1.5.7', 'EP' : 'BA.2.75.3.1.1.4', 'EU': 'XBB.1.5.26', 'EY': 'BQ.1.13.1.1.1', 'FB': 'BQ.1.2.1', 'FD': 'XBB.1.5.15', 'FE': 'XBB.1.18.1', 'FG': 'XBB.1.5.16', 'L': 'B.1.1.10', 'P': 'B.1.1.28', 'Q': 'B.1.1.7', 'R': 'B.1.1.316'}
	# 粗粒度条件下, 将较为流行的毒株合并
	if not fine_grained:
		alias_dict.update({'BF': 'BA.5.2.1', 'BN': 'BA.2.75.5', 'BQ': 'BA.5.3.1.1.1.1', 'CH': 'BA.2.75.3.4.1.1', 'DY': 'BA.5.2.48'})
	for lineage in alias_dict.keys():
		# "L".X -> "Expand L".X
		lineages[lineages.str.startswith(lineage + '.')] = alias_dict[lineage]
		
	# 合并谱系, 注意前缀关系
	# 合并谱系 (重组谱系 - Omicron 时代 - Delta 时代 - 前 Delta 时代)
	if fine_grained:
		lineage_list = ['XBB.1.16', 'XBB.1.22', 'XBB.1.5', 'XBB.1.9', 'XBB.2', 'XBF'] + ['BA.1', 'BA.2', 'BA.3', 'BA.4', 'BA.5', 'BF.7', 'BN', 'BQ.1', 'CH.1', 'DY'] + ['A', 'B.1.1.529', 'B.1.1.7', 'B.1.617.2']
	else:
		lineage_list = ['XBB.1', 'XBB.2'] + ['BA.5', 'BA.2.75'] + ['A', 'B.1.1.529', 'B.1.1.7', 'B.1.617.2']
	for lineage in lineage_list:
		# 不以 * 结尾, 即没有被合并过的
		lineages[(lineages.str.startswith(lineage + '.')) & (~lineages.str.endswith('*'))] = lineage + '*'
		lineages[lineages == lineage] = lineage + '*'
	
	# 合并谱系 (已经有子代被合并过的, 注意从底向上)
	if fine_grained:
		lineage_dict = {'XBB.1': 'XBB.1*', 'BF': 'BF*', 'B.1.1': 'B.1.1*', 'B.1': 'B.1*', 'B': 'B*'}
	else:
		lineage_dict = {'XBB': 'XBB*', 'BA': 'BA*', 'B.1.1': 'B.1.1*', 'B.1': 'B.1*', 'B': 'B*'}
	
	for lineage in lineage_dict.keys():
		# 替换
		lineages[(lineages.str.startswith(lineage + '.')) & (~lineages.str.endswith('*'))] = lineage_dict[lineage]
		lineages[lineages == lineage] = lineage_dict[lineage]

	# 覆盖
	dataframe['lineage'] = lineages

	# 分区间取出并统计
	end_date = date.today() if end_date == '' else date.fromisoformat(end_date)
	start_date = date.fromisoformat('2019-12-24') if start_date == '' else date.fromisoformat(start_date)
	query_start_date = date.fromisoformat('2019-12-24') if query_start_date == '' else date.fromisoformat(query_start_date)

	right = end_date
	left = right - timedelta(days = interval - 1)
	
	# (left, right) -> lineage_series
	interval_distribution_collection = {}
	interval_num_seq_collection = {}
	
	while right >= start_date:
		if left < start_date and left < query_start_date:
			left = start_date
		print('Processing: [', left, ',', right, ']')
		minCollectDate = str(left)
		maxCollectDate = str(right)
		
		# 取出一个区间的数据
		df = dataframe[(dataframe['collectDate'] >= minCollectDate) & (dataframe['collectDate'] <= maxCollectDate)].copy()
			
		num_seqs = len(df)
		if num_seqs > 0:
			lineages = df['lineage']
			# 合并其它占比小于 1% 的谱系
			for lineage in set(lineages):
				if (lineages == lineage).sum() < 0.01 * num_seqs:
					lineages[lineages == lineage] = 'Others (<1.0%)'		
			# 统计
			lineages = lineages.value_counts()
			# 增加
			interval_distribution_collection[right.strftime('%y-%m-%d')] = lineages.to_dict()
			interval_num_seq_collection[right.strftime('%y-%m-%d')] = num_seqs
		else:
			print('No data.')
		
		# 下一个区间
		right = right - timedelta(days = step)
		left = right - timedelta(days = interval - 1)
	
	# 日期区间 | 谱系1占比 | 谱系2占比 | ...
	data_lineages = pd.DataFrame.from_dict(interval_distribution_collection, orient = 'index')
	# 归一化
	data_lineages = data_lineages.fillna(0)
	data_lineages = data_lineages.div(data_lineages.sum(axis = 1), axis = 0)
	data_lineages = 100.0 * data_lineages
	data_lineages = data_lineages.sort_index()
	
	data_num_seqs = pd.DataFrame.from_dict(interval_num_seq_collection, orient = 'index')
	data_num_seqs = data_num_seqs.sort_index()

	print('Done.\n')
	return data_lineages, data_num_seqs

def visualize(region, data_lineages, data_num_seqs, fine_grained, interval = 28):
	n_colors = data_lineages.shape[1]
	sns.set_theme()
	sns.set_palette('Spectral', n_colors = n_colors)

	fig = plt.figure(figsize = (20, 10))
	ax = fig.add_subplot(111)

	ax = data_lineages.plot(kind = 'bar', ax = ax, stacked = True, ylim = (0.0, 100.0), width = 1.0, edgecolor = None, linewidth = 0.0)
	ax.set_ylabel('Lineage Proportion (%)')
	ax.tick_params(axis = 'x', which = 'major', labelsize = 8)
	ax.tick_params(axis = 'x', which = 'minor', labelsize = 6)
	# 共用 y 轴
	ax_seq = ax.twinx()
	ax_seq = data_num_seqs.plot(kind = 'line', ax = ax_seq, color = 'black')
	# ax_seq.set_yscale('log')
	ax_seq.set_ylabel('# Sequences')
	ax_seq.set_ylim(bottom = 0.0)
	# 对齐坐标轴线
	ax_seq.set_yticks(np.linspace(ax_seq.get_yticks()[0], ax_seq.get_yticks()[-1], len(ax.get_yticks())))
	# 隐藏格线
	ax.grid(False)
	ax_seq.grid(False)
	ax.legend(bbox_to_anchor = (-.05, 1.0), loc = 'upper right', borderaxespad = 0)
	ax_seq.get_legend().remove()

	plt.title('{} SARS-CoV-2 Lineage Distribution ({}, Bar = Sequences Collected in Previous {} Days)'.format('China (Mainland)' if region == 'ChinaMainland' else region, 'Fine Grained' if fine_grained else 'Coarse Grained', interval), fontsize = 18)
	plt.savefig('{}-{}.png'.format(region, 'fine' if fine_grained else 'coarse'), dpi = 300, bbox_inches = 'tight')
	
	plt.show()

def print_usage():
	print()
	print('Usage:')
	print('Get data from scratch:\n\tGET <region> [FROM <start_date>] [TO <end_date>] [INTERVAL <interval>]')
	print('Update data from cache file:\n\tUPDATE <region> [FROM <start_date>] [TO <end_date>] [INTERVAL <interval>]')
	print('Process lineages and visualize:\n\tVISUALIZE [FINE | COARSE] [FROM <start_date>] [TO <end_date>] [INTERVAL <interval>] [STEP <step>]')
	print('Get help:\n\tHELP')
	print('Exit:\n\tEXIT')
	print()


if __name__ == '__main__':
	print('+----------------------------------------------------------------+')
	print('|      SARS-CoV-2 Interactive Lineage Distribution Analysis      |')
	print('|                      Ver. 1.0.0, by G.Cui                      |')
	print('+----------------------------------------------------------------+')
	print_usage()
	
	data = None
	region = 'World'
	
	while True:
		input_cmd = re.split(r'[ ]+', input('>>> '))

		# 查询相关
		query_start_date = '2019-12-20'
		query_end_date = ''
		query_interval = 1

		# 数据划分/绘制相关
		data_interval = 14
		data_step = 7
		fine_grained = True
		visualize_start_date = '2020-01-01'
		visualize_end_date = ''

		data_lineages, data_num_seqs = None, None

		# 自动机解析
		state = 'START'
		# 逐个单词
		for token in input_cmd:
			# 起始状态
			if state == 'START':
				if token.upper() == 'EXIT':
					state = 'EXIT'
					break
				elif token.upper() == 'HELP':
					state = 'HELP'
					break
				if token.upper() in ['GET', 'UPDATE']:
					state = 'GOT_GET_OR_UPDATE'
				elif token.upper() == 'VISUALIZE':
					state = 'GOT_VISUALIZE'
				else:
					state = 'REJECT'
					break
			elif state == 'GOT_GET_OR_UPDATE':
				region = token
				state = 'GOT_REGION'
			elif state == 'GOT_REGION':
				if token.upper() in ['FROM', 'TO', 'INTERVAL']:
					state = 'WAIT_' + token.upper()
				else:
					state = 'REJECT'
					break
			elif state == 'WAIT_FROM':
				query_start_date = token
				state = 'GOT_FROM'
			elif state == 'GOT_FROM':
				if token.upper() in ['TO', 'INTERVAL']:
					state = 'WAIT_' + token.upper()
				else:
					state = 'REJECT'
					break
			elif state == 'WAIT_TO':
				query_end_date = token
				state = 'GOT_TO'
			elif state == 'GOT_TO':
				if token.upper() == 'INTERVAL':
					state = 'WAIT_INTERVAL'
				else:
					state = 'REJECT'
					break
			elif state == 'WAIT_INTERVAL':
				try:
					query_interval = int(token)
					state = 'GOT_INTERVAL'
				except:
					state = 'REJECT'
					break
			elif state == 'GOT_INTERVAL':
				state = 'REJECT'
				break
			elif state == 'GOT_VISUALIZE':
				if token.upper() in ['FINE', 'COARSE']:
					fine_grained = (token.upper() == 'FINE')
				elif token.upper() in ['FROM', 'TO', 'INTERVAL', 'STEP']:
					state = 'WAIT_VIS_' + token.upper()
				else:
					state = 'REJECT'
					break
			elif state == 'WAIT_VIS_FROM':
				visualize_start_date = token
				state = 'GOT_VIS_FROM'
			elif state == 'GOT_VIS_FROM':
				if token.upper() in ['TO', 'INTERVAL', 'STEP']:
					state = 'WAIT_VIS_' + token.upper()
				else:
					state = 'REJECT'
					break
			elif state == 'WAIT_VIS_TO':
				visualize_end_date = token
				state = 'GOT_VIS_TO'
			elif state == 'GOT_VIS_TO':
				if token.upper() == 'INTERVAL':
					state = 'WAIT_VIS_INTERVAL'
				else:
					state = 'REJECT'
					break
			elif state == 'WAIT_VIS_INTERVAL':
				try:
					data_interval = int(token)
					state = 'GOT_VIS_INTERVAL'
				except:
					state = 'REJECT'
					break
			elif state == 'GOT_VIS_INTERVAL':
				if token.upper() == 'STEP':
					state = 'WAIT_VIS_STEP'
				else:
					state = 'REJECT'
					break
			elif state == 'WAIT_VIS_STEP':
				try:
					data_step = int(token)
					state = 'GOT_VIS_STEP'
				except:
					state = 'REJECT'
					break

		if state == 'EXIT':
			break

		if state == 'HELP':
			print_usage()
		elif state == 'REJECT' or state not in ['GOT_REGION', 'GOT_VISUALIZE', 'GOT_FROM', 'GOT_TO', 'GOT_INTERVAL', 'GOT_STEP', 'GOT_VIS_FROM', 'GOT_VIS_TO', 'GOT_VIS_INTERVAL', 'GOT_VIS_STEP']:
			print('Invalid command!')
			print_usage()
			continue

		cmd = input_cmd[0].upper()
		if cmd == 'GET':
			data = get_data(region = region, start_date = query_start_date, end_date = query_end_date, query_interval = query_interval)
		elif cmd == 'UPDATE':
			data = update_data(region = region, query_interval = query_interval)
		elif cmd == 'VISUALIZE':
			data_lineages, data_num_seqs = process_lineage(data = data, fine_grained = fine_grained, start_date = visualize_start_date, end_date = visualize_end_date, interval = data_interval, step = data_step, query_start_date = query_start_date)
			visualize(region = region, data_lineages = data_lineages, data_num_seqs = data_num_seqs, fine_grained = fine_grained, interval = data_interval)			