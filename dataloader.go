package main

import (
	"encoding/binary"
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
	var magic, numImages, rows, cols int32
	binary.Read(f, binary.BigEndian, &magic)
	binary.Read(f, binary.BigEndian, &numImages)
	binary.Read(f, binary.BigEndian, &rows)
	binary.Read(f, binary.BigEndian, &cols)

	imgSize := int(rows * cols)
	images := make([][]float64, numImages)
	buf := make([]byte, imgSize)
	for i := range images {
		f.Read(buf)
		img := make([]float64, imgSize)
		for j, b := range buf {
			img[j] = float64(b) / 255.0 // normalize to [0,1]
		}
		images[i] = img
	}
	return images, nil
}

func readIdxLabelsFloat64(path string) ([]float64, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var magic, numLabels int32
	binary.Read(f, binary.BigEndian, &magic)
	binary.Read(f, binary.BigEndian, &numLabels)

	raw := make([]byte, numLabels)
	f.Read(raw)

	labels := make([]float64, numLabels)
	for i, b := range raw {
		labels[i] = float64(b)
	}
	return labels, nil
}
