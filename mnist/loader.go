// util transpiled from https://www.kaggle.com/code/hojjatk/read-mnist-dataset

package mnist

import (
	"encoding/binary"
	"fmt"
	"os"
)

func LoadMNIST(imagesTrainPath, labelsTrainPath, imagesTestPath, labelsTestPath string) (trainInputs [][]float64, trainLabels [][]float64, testInputs [][]float64, testLabels [][]float64, err error) {
	trainInputs, trainLabels, err = loadImagesAndLabels(imagesTrainPath, labelsTrainPath)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	testInputs, testLabels, err = loadImagesAndLabels(imagesTestPath, labelsTestPath)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	return trainInputs, trainLabels, testInputs, testLabels, nil
}

func loadImagesAndLabels(imagesPath, labelsPath string) (inputs [][]float64, labels [][]float64, err error) {
	imgData, err := os.ReadFile(imagesPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read images file: %w", err)
	}
	lblData, err := os.ReadFile(labelsPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read labels file: %w", err)
	}

	if len(imgData) < 16 {
		return nil, nil, fmt.Errorf("images file too small")
	}
	magic := binary.BigEndian.Uint32(imgData[0:4])
	if magic != 2051 {
		return nil, nil, fmt.Errorf("invalid images magic number: %d", magic)
	}
	numImages := int(binary.BigEndian.Uint32(imgData[4:8]))
	rows := int(binary.BigEndian.Uint32(imgData[8:12]))
	cols := int(binary.BigEndian.Uint32(imgData[12:16]))
	expectedSize := 16 + numImages*rows*cols
	if len(imgData) != expectedSize {
		return nil, nil, fmt.Errorf("images file size mismatch: got %d, expected %d", len(imgData), expectedSize)
	}

	if len(lblData) < 8 {
		return nil, nil, fmt.Errorf("labels file too small")
	}
	magic = binary.BigEndian.Uint32(lblData[0:4])
	if magic != 2049 {
		return nil, nil, fmt.Errorf("invalid labels magic number: %d", magic)
	}
	numLabels := int(binary.BigEndian.Uint32(lblData[4:8]))
	if numLabels != numImages {
		return nil, nil, fmt.Errorf("images (%d) and labels (%d) count mismatch", numImages, numLabels)
	}

	inputs = make([][]float64, numImages)
	pixelStart := 16
	for i := 0; i < numImages; i++ {
		inputs[i] = make([]float64, rows*cols)
		for j := 0; j < rows*cols; j++ {
			inputs[i][j] = float64(imgData[pixelStart]) / 255.0
			pixelStart++
		}
	}

	labels = make([][]float64, numLabels)
	labelStart := 8
	for i := 0; i < numLabels; i++ {
		lbl := int(lblData[labelStart])
		labels[i] = make([]float64, 10)
		labels[i][lbl] = 1.0
		labelStart++
	}

	return inputs, labels, nil
}
