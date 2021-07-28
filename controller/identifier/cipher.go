package identifier

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/pem"
	"errors"
	"io/ioutil"
	"os"
	"path/filepath"
)

const (
	RSA = "RSA_CIPHER"
	PRIVATE = "RSA PRIVATE KEY"
	PUBLIC = "PUBLIC KEY"
)

type Cipher struct {
	cipherType string
	privateKey *rsa.PrivateKey
	publicKey *rsa.PublicKey
	keyType string
}

func CreateNewCipher(cType string,keyType string)(*Cipher,error){
	// create new instance of the cipher
	var pb *rsa.PublicKey
	var pv *rsa.PrivateKey
	c:=Cipher{cipherType: cType,privateKey: pv,publicKey: pb,keyType: keyType}
	if cType == ""{
		err:=errors.New("cipher type can not be empty")
		return &c,err
	}

	return &c,nil
}


func (cp *Cipher)GenerateKey(output string)(map[string]string,error){
	// generate private and public key
	// check folder existence
	var keys map[string]string
	_,err := os.Stat(output)
	if os.IsNotExist(err){
		err = os.MkdirAll(output,0755)
		if err!=nil{
			return keys,err
		}
	}

	switch cp.cipherType {

	case RSA:
		// create the file paths
		privateKeyPath:= filepath.Join(output,"private.pem")
		publicKeyPath:=filepath.Join(output,"public.pem")


		// generate keys
		privateKey,err:= rsa.GenerateKey(rand.Reader,2048)
		if err!=nil{
			return keys,err
		}

		publicKey:=&privateKey.PublicKey


		// dump private key to file
		var privateKeyBytes []byte = x509.MarshalPKCS1PrivateKey(privateKey)
		privateKeyBlock:=&pem.Block{
			Type: PRIVATE,
			Bytes: privateKeyBytes,
		}

		privatePem,err:= os.Create(privateKeyPath)
		if err != nil{
			return keys,err
		}

		err = pem.Encode(privatePem,privateKeyBlock)
		if err!=nil{
			return keys,err
		}

		// dump public key to file
		publicKeyBytes,err:= x509.MarshalPKIXPublicKey(publicKey)
		if err!= nil{
			return keys,err
		}

		publicKeyBlock:=&pem.Block{
			Type: PUBLIC,
			Bytes: publicKeyBytes,
		}

		publicPem,err:=os.Create(publicKeyPath)
		if err!=nil{
			return keys,err
		}

		err = pem.Encode(publicPem,publicKeyBlock)
		if err!=nil{
			return keys,err
		}

		keys = make(map[string]string)
		keys["Private"] = privateKeyPath
		keys["Public"] = publicKeyPath

		return keys,nil

	default:

		err = errors.New("there is no more algorithm provided")
		return keys,err

	}

}

//func (cp *Cipher ) HasKey()bool{
//	// check that the cipher has the key or not
//	return cp.key!=""
//}

func (cp *Cipher)Init(keyPath string,keyType string)error{
	// check that the file is exists
	_,err:=os.Stat(keyPath)
	if err!=nil{
		return err
	}

	switch cp.cipherType {

	case RSA:
		keyIO,err := ioutil.ReadFile(keyPath)
		if err!=nil{
			return err
		}
		key,_:=pem.Decode(keyIO)

		if keyType == PRIVATE{
			cp.keyType = PRIVATE
			var privateParsedKey interface{}
			privateParsedKey,err:= x509.ParsePKCS1PrivateKey(key.Bytes)
			if err!=nil{
				return err
			}
			var privateKey *rsa.PrivateKey
			var ok bool
			privateKey,ok = privateParsedKey.(*rsa.PrivateKey)
			if !ok{
				err = errors.New("private key can`t parse")
				return err
			}
			cp.privateKey = privateKey
			return nil
		}else{
			cp.keyType = PUBLIC
			var publicParsedKey interface{}
			publicParsedKey,err:=x509.ParsePKIXPublicKey(key.Bytes)
			if err!=nil{
				return err
			}
			var publicKey *rsa.PublicKey
			var ok bool
			publicKey,ok = publicParsedKey.(*rsa.PublicKey)
			if !ok{
				err = errors.New("public key can`t parse")
				return err
			}
			cp.publicKey = publicKey
			return nil
		}

	default:
		err = errors.New("there is no more algorithm provided")
		return err
	}
}

func (cp *Cipher)Type()string{
	// get type of cipher
	return cp.keyType
}

func (cp *Cipher)PrivateKey()*rsa.PrivateKey{
	// return private key
	return cp.privateKey
}

func (cp *Cipher)PublicKey()*rsa.PublicKey{
	// return public key
	return cp.publicKey
}


func (cp *Cipher)Encrypt(msg string)([]byte,error){
	// encrypt the message
	encryptedBytes,err:=rsa.EncryptOAEP(
		sha256.New(),
		rand.Reader,
		cp.publicKey,
		[]byte(msg),
		nil)
	if err!=nil{
		return encryptedBytes,err
	}
	return encryptedBytes,nil
}


func (cp *Cipher)Decrypt(enByte []byte)([]byte,error){
	// decrypt the message
	decryptedBytes, err := cp.privateKey.Decrypt(nil, enByte, &rsa.OAEPOptions{Hash: crypto.SHA256})
	if err != nil {
		return decryptedBytes,err
	}
	return decryptedBytes,err
}

