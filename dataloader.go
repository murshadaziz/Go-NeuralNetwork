package main

import (
	"bufio"
	"encoding/binary"
	"io"
	"os"
)

// Reads 28x28 images from the .idx file and return a 2d slice (60000x784) where each row represents one image
func readIdxImagesFloat64(path string) ([][]float64, error) {
	// os.Open() takes file path as input and returns *os.file that is a standard pointer to file structure
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	// func Read(r io.Reader, order ByteOrder, data any) error
	// binary.Read takes anything that implements the io.Reader interface and as *os.file implements
	// the read method in that interface
	// 2nd parameter is the ByteOrder BigEndian means that the if the method is reading 4 bytes then
	// the most significant digit will be on the left
	// 3rd parameter is the size of bytes it will read and store, here it will read uint32 = 4 bytes
	// So binary.Read takes *os.file, ByteOrder and the data to store in as input interpretes the bytes
	// as the datatype of the input data and stores in it
	// .idx3 file has a 4 byte header so we read 4 bytes

	// makes a wrapper buffer for *os.file to reduce the syscalls
	// (bufio.NewReader wraps anything that implements the io.Reader interface that has a Read method)
	// (and returns a *bufio.Reader that also implements the io.Reader interface)
	reader := bufio.NewReader(f)
	var magic, numImages, rows, cols uint32
	// err handling after each Read operation
	// Magic number is the file format e.g it has value 00000803 for idx3 files with unsigned bytes of 3d data (28x28x60000)
	err = binary.Read(reader, binary.BigEndian, &magic)
	if err != nil {
		return nil, err
	}
	// Next 4 bytes represent the number of images
	err = binary.Read(reader, binary.BigEndian, &numImages)
	if err != nil {
		return nil, err
	}
	// represent the number of rows per image
	err = binary.Read(reader, binary.BigEndian, &rows)
	if err != nil {
		return nil, err
	}
	// represent the number of columns per image
	err = binary.Read(reader, binary.BigEndian, &cols)
	if err != nil {
		return nil, err
	}

	// rows*colums represent the size of image e.g 28x28 = 728
	imgSize := int(rows * cols)
	// makes a 2d slice with rows = numImages
	images := make([][]float64, int(numImages))
	// makes a temporary buffer having size equal to the size of image to read one image at a time
	buf := make([]byte, imgSize)
	// iterates through the all the images one by one
	for i := range images {
		// reads raw bytes into the buffer
		// ReadFull fills the buf and returns n = number of byters and err, n is discarded here
		_, err = io.ReadFull(reader, buf)
		if err != nil {
			return nil, err
		}
		// makes a 1d slice of size 28x28 for one image
		img := make([]float64, imgSize)
		// iterates through the buffer and takes out one byte at a time
		for j, b := range buf {
			// converts byte into float64 and stores in the 1d img slice
			img[j] = float64(b) / 255.0 // normalize to [0,1]
		}
		// assigns to the 2d slice as a row
		images[i] = img
	}
	// returns the 2d slice containing no of rows = numImages and number of columns = imgSize
	return images, nil
}

// Reads labels from .idx1 file and returns a 1d slice containing 1 label for each image e.g 60000 labels
func readIdxLabelsFloat64(path string) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	// makes a wrapper buffer for *os.file to reduce the syscalls
	// (bufio.NewReader wraps anything that implements the io.Reader interface that has a Read method)
	// (and returns a *bufio.Reader that also implements the io.Reader interface)
	reader := bufio.NewReader(f)
	// idx1 file has a 8 byte header so we read 8 bytes
	var magic, numLabels uint32
	// first four bytes represent the magic number 00000801 which tell that file has unsigned bytes in 1 dimension
	err = binary.Read(reader, binary.BigEndian, &magic)
	if err != nil {
		return nil, err
	}
	// next four bytes tells the number of labels
	err = binary.Read(reader, binary.BigEndian, &numLabels)
	if err != nil {
		return nil, err
	}
	// makes temporary buffer which stores the bytes of the all the labels e.g 60000
	raw := make([]byte, numLabels)
	// reads all the bytes into the raw byte buffer
	_, err = io.ReadFull(reader, raw)
	if err != nil {
		return nil, err
	}
	// makes a float64 1d slice of size equal to number of labels
	labels := make([]int, int(numLabels))
	// iterates over raw buffer taking out one byte at a time
	for i, b := range raw {
		// converts byte into float64 and stores in labels slice
		labels[i] = int(b)
	}
	// returns the 1d float64 slice
	return labels, nil
}
