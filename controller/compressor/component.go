package compressor

import (
	log "../logging"
	"os"
	"path/filepath"
)

var (
	folderInfor *os.FileInfo
	err error
)


func Zip(filename string,outputPath string, targetPath string){
	_,err:=os.Stat(targetPath)
	if os.IsNotExist(err){
		log.Println(log.DANG,"$ Target folder path is not exists")
	}

	_,err=os.Stat(outputPath)
	if os.IsNotExist(err){
		err=os.Mkdir(outputPath,0755)

		if err!=nil {
			log.Println(log.DANG, "$ Can`nt create folder path")
		}else{
			log.Println(log.SUCCESS,"$ Create folder path")
		}
	}

	fullFilePath := filepath.Join(outputPath,filename)

	log.Println(log.SUCCESS,fullFilePath)

	//r,err := zip.OpenReader(targetPath)
	//if err!=nil{
	//	log.Println(log.DANG,"$ Problem in reading source")
	//}
	//
	//defer r.Close()




}