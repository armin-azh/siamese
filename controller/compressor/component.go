package compressor

import (
	log "../logging"
	"archive/zip"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
)

var (
	folderInfor *os.FileInfo
	err error
)


func Zip(filename string,outputPath string, targetPath string)error{

	// check the target file folder existence
	_,err:=os.Stat(targetPath)
	if os.IsNotExist(err){
		log.Println(log.DANG,"$ Target folder path is not exists")
	}

	// check output folder existence, if not, create it
	_,err=os.Stat(outputPath)
	if os.IsNotExist(err){
		err=os.Mkdir(outputPath,0755)
		if err!=nil {
			return err
		}
	}

	// create full path of output
	fullFilePath := filepath.Join(outputPath,filename)

	archive,err:= os.Create(fullFilePath)

	if err!=nil{
		return err
	}

	defer archive.Close()


	// perform zip
	zipwriter:=zip.NewWriter(archive)

	defer zipwriter.Close()

	f1,err:=os.Open(targetPath)
	if err!=nil{
		return err
	}

	defer f1.Close()

	fmt.Println(f1)

	var files[]string

	err = filepath.Walk(targetPath, func(path string, info fs.FileInfo, err error) error {
		files = append(files, path)
		return nil
	})


	for _,file := range files{
		if filepath.Ext(file)!=""{
			f1, err := os.Open(file)
			if err != nil {
				return err
			}

			w, err := zipwriter.Create(file)

			if _,err:=io.Copy(w,f1);err!=nil{
				return err
			}

		}
	}

	return nil

}