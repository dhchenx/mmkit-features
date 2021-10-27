from mmkfeatures.fusion import log
from mmkfeatures.fusion.metadataconfigs import *
from tqdm import tqdm

#this function checks the heirarchy format of a given computatioanl sequence data. This will crash the program if data is in wrong format. If in correct format the return value is simply True 
def validate_data_format(data,root_name,verbose=True):
	log.status("Checking the format of the data in <%s> computational sequence ..."%root_name)

	pbar = log.progress_bar(total=len(data.keys()),unit=" Computational Sequence Entries",leave=False)
	failure=False
	if (type(data) is not dict):
		#this will cause the rest of the pipeline to crash - RuntimeError
		log.error("%s computational sequence data is not in heirarchy format ...",error=True)
	try:

		#for each video check the shapes of the intervals and features
		for vid in data.keys():
			if "F" in data[vid].keys():
				# check the intervals first - if failure simply show a warning - no exit since we want to identify all the cases
				if (len(data[vid]["ITV"]) != 0 and len(data[vid]["ITV"].shape) != 2):
					if verbose: log.error(
						"Timestamped content (e.g. video,audio) <%s> in  <%s> computational sequence has wrong intervals array shape. " % (
						vid, root_name), error=False)
					failure = True
				# check the features next
				if (len(data[vid]["F"]) != 0 and len(data[vid]["F"].shape) != 2):
					if verbose: log.error(
						"Timestamped content (e.g. video,audio)  <%s> in  <%s> computational sequence has wrong features array shape. " % (
						vid, root_name), error=False)
					failure = True
			else:
				# check the intervals first - if failure simply show a warning - no exit since we want to identify all the cases
				if (len(data[vid]["intervals"]) != 0 and len(data[vid]["intervals"].shape) != 2):
					if verbose: log.error(
						"Timestamped content (e.g. video,audio) <%s> in  <%s> computational sequence has wrong intervals array shape. " % (
						vid, root_name), error=False)
					failure = True
				# check the features next
				if (len(data[vid]["features"]) != 0 and len(data[vid]["features"].shape) ==0):
					if verbose: log.error(
						"Timestamped content (e.g. video,audio)  <%s> in  <%s> computational sequence has wrong features array shape. " % (
						vid, root_name), error=False)
					failure = True
			pbar.update(1)
	#some other thing has happened! - RuntimeError
	except:
		if verbose:
			log.error("<%s> computational sequence data format could not be checked. "%root_name,error=True)
		pbar.close()
	pbar.close()

	#failure during intervals and features check
	if failure:
		log.error("<%s> computational sequence data integrity check failed due to inconsistency in intervals and features. "%root_name,error=True)
	else:
		log.success("<%s> computational sequence data in correct format." %root_name)
	return True


#this function checks the computatioanl sequence metadata. This will not crash the program if metadata is missing. If metadata is there the return value is simply True 
def validate_metadata_format(metadata,root_name,verbose=True):
	log.status("Checking the format of the metadata in <%s> computational sequence ..."%root_name)
	failure=False
	if type(metadata) is not dict:
		log.error("<%s> computational sequence metadata is not key-value pairs!", error=True)
	presenceFlag=[mtd in metadata.keys() for mtd in featuresetMetadataTemplate]
	#check if all the metadata is set
	if all (presenceFlag) is False:
		#verbose one is not set
		if verbose:
			missings=[x for (x,y) in zip (featuresetMetadataTemplate,presenceFlag) if y is False]
			log.error("Missing metadata in <%s> computational sequence: %s"%(root_name,str(missings)),error=False)
		failure=True
	if failure:
		log.error(msgstring="<%s> computational sequence does not have all the required metadata ... continuing "%root_name,error=False)
		return False
	else:
		log.success("<%s> computational sequence metadata in correct format."%root_name)
		return True


