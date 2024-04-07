If you have Model already trained and ModelBuilder is not showing that,
you need to manually edit Model.mbconfig file as follows:

	1.	set "FolderPath" to non relative path of csfd_bin.tsv or csfd_bin_extremes.tsv file in the Shared project
	2.	set "ModelFilePath" to non relative path of Model.mlnet file in this project
