package compressor

import (
	log "../logging"
	"archive/zip"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

var (
	folderInfor *os.FileInfo
	err         error
)

func Zip(filename string, outputPath string, targetPath string) (string,error) {
	// function for zipping a given root
	// check the target file folder existence
	_, err := os.Stat(targetPath)
	if os.IsNotExist(err) {
		log.Println(log.DANG, "$ Target folder path is not exists")
	}

	// check output folder existence, if not, create it
	_, err = os.Stat(outputPath)
	if os.IsNotExist(err) {
		err = os.Mkdir(outputPath, 0755)
		if err != nil {
			return "",err
		}
	}

	// create full path of output
	fullFilePath := filepath.Join(outputPath, filename)

	archive, err := os.Create(fullFilePath)

	if err != nil {
		return "",err
	}

	defer archive.Close()

	// perform zip
	zipwriter := zip.NewWriter(archive)

	defer zipwriter.Close()

	f1, err := os.Open(targetPath)
	if err != nil {
		return "",err
	}

	defer f1.Close()

	var files []string

	err = filepath.Walk(targetPath, func(path string, info fs.FileInfo, err error) error {
		files = append(files, path)
		return nil
	})

	for _, file := range files {
		if filepath.Ext(file) != "" {
			f1, err := os.Open(file)
			if err != nil {
				return "",err
			}
			w, err := zipwriter.Create(file)
			if _, err := io.Copy(w, f1); err != nil {
				return "",err
			}

		}
	}

	return "",nil
}


func Unzip(target string,outputPath string)([]string,error){
	// function for unzipping a file
	// check directory existence

	var filenames []string
	_,err:= os.Stat(target)
	if os.IsNotExist(err){
		return filenames,err
	}

	_,err = os.Stat(outputPath)
	if os.IsNotExist(err){
		return filenames,err
	}

	reader,err:=zip.OpenReader(target)
	if err!=nil{
		return filenames,err
	}

	defer reader.Close()

	for _,file:= range reader.File{
		fPath:= filepath.Join(outputPath,file.Name)

		if !strings.HasPrefix(fPath,filepath.Clean(outputPath)+string(os.PathSeparator)){
			return filenames,fmt.Errorf("%s: illegal file path",fPath)
		}

		filenames = append(filenames,fPath)

		if file.FileInfo().IsDir(){
			os.MkdirAll(fPath,os.ModePerm)
			continue
		}

		// Make file
		if err = os.MkdirAll(filepath.Dir(fPath),os.ModePerm);err!=nil{
			return filenames,err
		}

		outFile, err := os.OpenFile(fPath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, file.Mode())
		if err != nil {
			return filenames, err
		}

		rc, err := file.Open()
		if err != nil {
			return filenames, err
		}

		_, err = io.Copy(outFile, rc)

		outFile.Close()
		rc.Close()

		if err != nil {
			return filenames, err
		}

	}

	return filenames, nil
}
