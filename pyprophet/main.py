# encoding: latin-1

# openblas + multiprocessing crashes for OPENBLAS_NUM_THREADS > 1 !!!
import os
import warnings
os.putenv("OPENBLAS_NUM_THREADS", "1")

try:
	profile
except NameError:
	profile = lambda x: x

from pyprophet import PyProphet, HolyGostQuery
from classifiers import SGDLearner, LDALearner, LinearSVMLearner, RbfSVMLearner, PolySVMLearner, LogitLearner
from semi_supervised import StandardSemiSupervisedTeacher

from config import standard_config, fix_config_types
from report import save_report, export_mayu
import sys
import time
import warnings
import logging
import cPickle
import zlib

#import pandas as pd
import numpy as np

def print_help():
	print
	script = os.path.basename(sys.argv[0])
	print "usage:"
	print "	   %s [options] input_file" % script
	print "   or "
	print "	   %s --help" % script
	print "   or "
	print "	   %s --version" % script
	CONFIG, info = standard_config()
	dump_config_info(CONFIG, info)


def print_version():
	import version
	print "%d.%d.%d" % version.version


def dump_config_info(config, info):
	print
	print "parameters:"
	print
	for k, v in sorted(config.items()):
		comment = info.get(k, "")
		print "	--%-40s   default: %-5r %s" % (k, v, comment)
	print


def dump_config(config):
	print
	print "used parameters:"
	print
	for k, v in sorted(config.items()):
		print "	%-40s   : %r" % (k, v)
	print


def main():
	_main(sys.argv[1:])


def _main(args):

	options = dict()
	path = None
	
	print "PyProphet, unified edition
	
	if "--help" in args:
		print_help()
		return

	if "--version" in args:
		print_version()
		return

	for arg in args:
		if arg.startswith("--"):
			if "=" in arg:
				pre, __, post = arg.partition("=")
				options[pre[2:]] = post
			else:
				options[arg[2:]] = True
		else:
			if path is not None:
				print_help()
				raise Exception("duplicate input file argument")
			path = arg

	if path is None:
		print_help()
		raise Exception("no input file given")

	CONFIG, info = standard_config()
	CONFIG.update(options)
	fix_config_types(CONFIG)
	dump_config(CONFIG)

	delim_in = CONFIG.get("delim.in", ",")
	delim_out = CONFIG.get("delim.out", ",")

	dirname = CONFIG.get("target.dir", None)
	if dirname is None:
		dirname = os.path.dirname(path)

	basename = os.path.basename(path)
	prefix, __ = os.path.splitext(basename)

	persisted_scorer = None
	apply_scorer = CONFIG.get("apply_scorer")
	if apply_scorer:
		if not os.path.exists(apply_scorer):
			raise Exception("scorer file %s does not exist" % apply_scorer)
		try:
			persisted_scorer = cPickle.loads(zlib.decompress(open(apply_scorer, "rb").read()))
		except:
			import traceback
			traceback.print_exc()
			raise

	apply_existing_scorer = persisted_scorer is not None

	class Paths(dict):
		def __init__(self, prefix=prefix, dirname=dirname, **kw):
			for k, postfix in kw.items():
				self[k] = os.path.join(dirname, prefix + postfix)
		__getattr__ = dict.__getitem__

	paths = Paths(scored_table="_with_dscore.csv",
					filtered_table="_with_dscore_filtered.csv",
					output="_output.csv",
					final_stat="_full_stat.csv",
					summ_stat="_summary_stat.csv",
					report="_report.pdf",
					cutoffs="_cutoffs.txt",
					svalues="_svalues.txt",
					d_scores_top_target_peaks="_dscores_top_target_peaks.txt",
					d_scores_top_decoy_peaks="_dscores_top_decoy_peaks.txt",
					mayu_cutoff="_mayu.cutoff",
					mayu_fasta="_mayu.fasta",
					mayu_csv="_mayu.csv",
					)

	if not apply_existing_scorer:
		pickled_scorer_path = os.path.join(dirname, prefix + "_scorer.bin")

	if not apply_existing_weights:
		trained_weights_path = os.path.join(dirname, prefix + "_weights.txt")

	if not CONFIG.get("target.overwrite", False):
		found_existing_file = False
		to_check = list(paths.keys())
		if not apply_existing_scorer:
			to_check.append(pickled_scorer_path)
		if not apply_existing_weights:
			to_check.append(trained_weights_path)
		for p in to_check:
			if os.path.exists(p):
				found_existing_file = True
				print "ERROR: %s already exists" % p
		if found_existing_file:
			print
			print "please use --target.overwrite option"
			print
			return

	format_ = "%(levelname)s -- [pid=%(process)s] : %(asctime)s: %(message)s"
	logging.basicConfig(level=logging.INFO, format=format_)
	logging.info("config settings:")
	for k, v in sorted(CONFIG.items()):
		logging.info("	%s: %s" % (k, v))
	start_at = time.time()
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		
		classifierType = CONFIG.get("classifier.type")
		if classifierType == "LDA":
			classifier = LDALearner
		elif classifierType == "SGD":
			classifier = SGDLearner
		elif classifierType == "linSVM":
			classifier = LinearSVMLearner
		elif classifierType == "rbfSVM":
			classifier = RbfSVMLearner
		elif classifierType == "polySVM":
			classifier = PolySVMLearner
		elif classifierType == "logit":
			classifier = LogitLearner
		else:
			print
			print "classifier '%s' is not supported" % classifierType
			print
			return
		
		method = HolyGostQuery(StandardSemiSupervisedTeacher(classifier))
		result_tables, clfs_df, needed_to_persist, trained_weights = method.process_csv(path, delim_in, persisted_scorer, persisted_weights)
		#(summ_stat, final_stat, scored_table) = pyp_res
		#(summ_statT, final_statT, scored_tableT) = true_res
	
	needed = time.time() - start_at

	train_frac 	= CONFIG.get("train.fraction")
	def printSumTable(str):
		if str in result_tables:
			with warnings.catch_warnings():
    				warnings.filterwarnings("ignore",category=DeprecationWarning)
				df = result_tables[str][0]
				if df is not None:
					print str
					print df[df.qvalue < 0.21][['qvalue', 'TP', 'cutoff']]
	print
	print "=" * 78
	print "%d%% of data used for training" % (train_frac*100)
	print "'" * 78
	print
	for k in result_tables.iterkeys():
		printSumTable(k)
	print
	print "=" * 78
	print
	
	if not CONFIG.get("no.file.output") and "res" in result_tables:
		summ_stat, final_stat, scored_table = result_tables['res']
		#if 'true_normal' in result_tables:
		#	summ_statT, final_statT, scored_tableT = result_tables['true_normal']
		#	summ_stat.to_csv(paths.summ_stat, sep=delim_out, index=False)
		#	print "WRITTEN: ", paths.summ_stat
		#	plot_data = save_report(paths.reportT, basename, scored_tableT, final_statT)
		#	print "WRITTEN: ", paths.report
		if summ_stat is not None:
			summ_stat.to_csv(paths.summ_stat, sep=delim_out, index=False)
			print "WRITTEN: ", paths.summ_stat

		if final_stat is not None:
			plot_data = save_report(paths.report, basename, scored_table, final_stat)
			print "WRITTEN: ", paths.report
			
			if CONFIG.get("all.output"):
				final_stat.to_csv(paths.final_stat, sep=delim_out, index=False)
				print "WRITTEN: ", paths.final_stat
				
				cutoffs, svalues, qvalues, top_target, top_decoys = plot_data
				for (name, values) in [("cutoffs", cutoffs), ("svalues", svalues), ("qvalues", qvalues),
								   ("d_scores_top_target_peaks", top_target),
								   ("d_scores_top_decoy_peaks", top_decoys)]:
					path = paths[name]
					with open(path, "w") as fp:
						fp.write(" ".join("%e" % v for v in values))
					print "WRITTEN: ", path
		
		if clfs_df is not None and CONFIG.get("all.output"):
			clfs_df.to_csv("clfs.csv", sep=delim_out, index=False)
			print "WRITTEN: ", "clfs.csv"
		
		scored_table.to_csv(pathes.scored_table, sep=delim_out, index=False)
		print "WRITTEN: ", paths.scored_table

		output = scored_table.rename(columns = {"d_score" : "pyProph_score", "m_score" : "qvalue"})
		output.to_csv(paths.output, sep=delim_out, index=False)
		print "WRITTEN: ", paths.output

		filtered_table = scored_table[scored_table.d_score > CONFIG.get("d_score.cutoff")]
		filtered_table.to_csv(paths.filtered_table, sep=delim_out, index=False)
		print "WRITTEN: ", paths.filtered_table
		
		if not apply_existing_scorer and CONFIG.get("all.output"):
			bin_data = zlib.compress(cPickle.dumps(needed_to_persist, protocol=2))
			with open(pickled_scorer_path, "wb") as fp:
				fp.write(bin_data)
			print "WRITTEN: ", pickled_scorer_path

		if not apply_existing_weights:
			np.savetxt(trained_weights_path,trained_weights,delimiter="\t")
			print "WRITTEN: ", trained_weights_path

		if CONFIG.get("export.mayu", True):
			export_mayu(paths.mayu_cutoff, paths.mayu_fasta, paths.mayu_csv, scored_table, final_stat)
			print "WRITTEN: ", paths.mayu_cutoff
			print "WRITTEN: ", paths.mayu_fasta
			print "WRITTEN: ", paths.mayu_csv
		print

	seconds = int(needed)
	msecs 	= int(1000 * (needed - seconds))
	minutes = int(needed / 60.0)
	hours 	= int(minutes / 60.0)

	print "NEEDED %02d:%02d:%02d.%03d wall time" % (hours, minutes, seconds, msecs)
	print
