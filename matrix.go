// Package linalg implements matrix type and operations to operate with linear algebra
package linalg

import (
	"errors"
	"fmt"
	"strings"
)

var empty_matrix = &Matrix{
	data:    [][]float64{},
	lines:   0,
	columns: 0,
}

// Matrix intends to represent a matrix with its rows and columns
// It is possible to create a new matrix by using [NewMatrix] method and providing an bidimensional array of float64 like:
//
//	[][]float64 {
//	  {1, 2, 3, 4},
//	  {4, 3, 2, 1}
//	}
//
// It is important to notice that the above matrix has 2 lines and 4 colunms
type Matrix struct {
	lines   int
	columns int
	data    [][]float64
}

// NullMatrix returns a pointer to a null matrix.
// NullMatrix are square and filled with zeros
// This is just a handy method to facilitate the calculations when a null matrix is needed
// As operations on matrix always generate a new one, there is no problem in returning a pointer here
func NullMatrix(size int) *Matrix {
	data := make([][]float64, size)

	for i := range data {
		data[i] = make([]float64, size)
	}

	return &Matrix{
		data:    data,
		lines:   size,
		columns: size,
	}
}

func NewIdentityMatrix(size int) *Matrix {
	data := make([][]float64, size)

	for i := range data {
		data[i] = make([]float64, size)
		data[i][i] = 1
	}

	return &Matrix{
		data:    data,
		lines:   size,
		columns: size,
	}
}

// NewMatrix creates matrix from a bidimensional float64 array and return its pointer.
// The format to use when creating the array is as follows:
//
//	[][]float64 {
//	  {1, 2, 3, 4},
//	  {4, 3, 2, 1}
//	}
//
// It is important to notice that the above matrix has 2 lines and 4 colunms
func NewMatrix(data [][]float64) *Matrix {
	m := &Matrix{
		data: data,
	}

	// checking null matrix
	if len(data) == 0 {
		return empty_matrix
	}

	m.lines = len(data)

	if len(data[0]) == 0 {
		return empty_matrix
	}

	m.columns = len(data[0])
	return m
}

// Lines returns the quantity of lines of the matrix
func (m *Matrix) Lines() int {
	return m.lines
}

// Columns returns the quantity of columns of the matrix
func (m *Matrix) Columns() int {
	return m.columns
}

// Position returns a value in the specified row and column of the matrix.
// Opposite to the programming notation, mathematical one uses 1-based indexes.
func (m *Matrix) Position(row int, col int) (float64, error) {
	if m.lines < row {
		return 0, errors.New("row number is higher than the matrix's lines")
	}

	if m.columns < col {
		return 0, errors.New("column number is higher than the matrix's columns")
	}

	if row == 0 {
		return 0, errors.New("invalid row")
	}

	if col == 0 {
		return 0, errors.New("invalid column")
	}

	return m.data[row-1][col-1], nil
}

// Sum returns a new matrix that is the result of the sum of the underlying matrix with the one passed as parameter
// The matrix need to have the same size or an error will be returned
func (m *Matrix) Sum(B *Matrix) (*Matrix, error) {
	if m.lines != B.lines || m.columns != B.columns {
		return nil, errors.New("Matrix size is not the same")
	}

	result := make([][]float64, m.lines)

	for i := range result {
		result[i] = make([]float64, m.columns)
		for j := range result[i] {
			result[i][j] = m.data[i][j] + B.data[i][j]
		}
	}
	return NewMatrix(result), nil
}

// Sub returns a new matrix that is the result of the subtraction of the underlying matrix with the one passed as parameter
// The matrix need to have the same size or an error will be returned
func (m *Matrix) Sub(B *Matrix) (*Matrix, error) {
	if m.lines != B.lines || m.columns != B.columns {
		return nil, errors.New("Matrix size is not the same")
	}

	result := make([][]float64, m.lines)

	for i := range result {
		result[i] = make([]float64, m.columns)
		for j := range result[i] {
			result[i][j] = m.data[i][j] - B.data[i][j]
		}
	}
	return NewMatrix(result), nil
}

// ScalarProduct multiplies the matrix by an scalar float and returns the new resulting matrix
func (m *Matrix) ScalarProduct(scalar float64) *Matrix {
	result := make([][]float64, m.lines)

	for i := range result {
		result[i] = make([]float64, m.columns)
		for j := range result[i] {
			result[i][j] = m.data[i][j] * scalar
		}
	}
	return NewMatrix(result)
}

// Transpose returns a new matrix that is the transposed matrix of the original one.
// The lines of the original matrix becomes the columns of the transposed and the columns of the original becomes the lines of the transposed
func (m *Matrix) Transpose() *Matrix {
	result := make([][]float64, m.columns)
	for i := range result {
		result[i] = make([]float64, m.lines)
		for j := range result[i] {
			result[i][j] = m.data[j][i]
		}
	}
	return NewMatrix(result)
}

// Product multiplies the matrix by another on ans returns a new resulting matrix
// It also checks if the matrixes can be multiplied and if not, returns an error
func (m *Matrix) Product(B *Matrix) (*Matrix, error) {
	if m.columns != B.Lines() {
		return nil, errors.New("matrix B number of lines id different of matrix A number of columns")
	}

	data := make([][]float64, m.lines)
	for i := range data {
		data[i] = make([]float64, B.columns)
		for j := range data[i] {
			for k := 0; k < m.Columns(); k++ {
				data[i][j] += m.data[i][k] * B.data[k][j]
			}
		}
	}
	return NewMatrix(data), nil
}

// LU decomposes the matrix in an lower (L) and upper (U) triangular matrix in a way that A = LU
// This method uses the Doolittle algorithm
// Also, it requires a square matrix
func (m *Matrix) LU() (L *Matrix, U *Matrix, err error) {
	if m.lines != m.columns {
		return nil, nil, errors.New("matrix can't be decomposed, number of lines and columns are different")
	}

	l_data := make([][]float64, m.lines)
	u_data := make([][]float64, m.lines)

	// Initialize l_data and u_data
	for i := range l_data {
		l_data[i] = make([]float64, m.lines)
		u_data[i] = make([]float64, m.lines)
	}

	for i := 0; i < m.lines; i++ {

		// upper triangular
		for k := 0; k < m.lines; k++ {

			// summation of L(i,j) * U (j,k)
			sum := 0.
			for j := 0; j < i; j++ {
				sum += l_data[i][j] * u_data[j][k]
			}

			// Evaluating U(i,k)
			u_data[i][k] = m.data[i][k] - sum
		}

		// Lower Triangular
		for k := 0; k < m.lines; k++ {
			if i == k {
				l_data[i][i] = 1. // Diagonal as 1
			} else {
				// Summation of L(k,j) * U(j,i)
				sum := 0.
				for j := 0; j < i; j++ {
					sum += l_data[k][j] * u_data[j][i]
				}

				// Evaluating L(k,i)
				if u_data[i][i] == 0 {
					l_data[k][i] = 0
				} else {
					l_data[k][i] = (m.data[k][i] - sum) / u_data[i][i]
				}
			}
		}
	}

	return NewMatrix(l_data), NewMatrix(u_data), nil
}

// Determinant returns de determinant value of a square matrix or an error if it cannot be calculated
// It uses the method of LU Decomposition (Doolittle) to achieve the goal for a n x n matrix
func (m *Matrix) Determinant() (float64, error) {
	if m.lines == 0 || m.columns == 0 {
		return 0, errors.New("empty matrix")
	}

	if m.lines != m.columns {
		return 0, errors.New("matrix must be square")
	}

	if m.lines == 1 {
		return m.data[0][0], nil
	}

	if m.lines == 2 {
		return m.data[0][0]*m.data[1][1] - m.data[0][1]*m.data[1][0], nil
	}

	_, U, _ := m.LU()
	diag := 1.
	for i := 0; i < U.lines; i++ {
		diag *= U.data[i][i]
	}
	return diag, nil
}

// Equals determines if the underlying matrix is equal to another one.
// By Equals, is considered that every value is equal to another one in the other matrix
func (m *Matrix) Equals(B *Matrix) bool {
	if m.lines != B.lines || m.columns != B.columns {
		return false
	}

	for i := 0; i < m.lines; i++ {
		for j := 0; j < m.columns; j++ {
			if m.data[i][j] != B.data[i][j] {
				return false
			}
		}
	}

	return true
}

// Inverse returns a new matrix being the inverse form of the original one
// If the matrix os not square it returns an error
// If the determinant of the matrix is 0, there is no inverse for the matrix and an error will be returned
func (m *Matrix) Inverse() (*Matrix, error) {
	if m.lines != m.columns {
		return nil, errors.New("matrix musts be square")
	}

	det, err := m.Determinant()
	if err != nil {
		return nil, err
	}

	if det == 0 {
		return nil, errors.New("determinant is zero. Matrix cannot be inverted")
	}

	/*
			1. Form the augmented matrix by the identity matrix.
		  2. Perform the row reduction operation on this augmented matrix to generate a row reduced echelon form of the matrix.
		  3. The following row operations are performed on augmented matrix when required:
		     - Interchange any two row.
		     - Multiply each element of row by a non-zero integer.
		     - Replace a row by the sum of itself and a constant multiple of another row of the matrix.
	*/

	order := m.lines
	augmented := make([][]float64, order)
	// copy m.data to augmented
	for i := range augmented {
		augmented[i] = make([]float64, order*2)
		for j := 0; j < order; j++ {
			augmented[i][j] = m.data[i][j]
		}
	}

	for i := 0; i < order; i++ {

		// create augmented matrix
		for j := 0; j < order*2; j++ {
			// add 1 to diagonal places of augmented part
			if j == (i + order) {
				augmented[i][j] = 1
			}
		}
	}

	// Interchange the row of matrix, starting in last row
	for i := order - 1; i > 0; i-- {
		if augmented[i-1][0] < augmented[i][0] {
			tempSlice := augmented[i]
			augmented[i] = augmented[i-1]
			augmented[i-1] = tempSlice
		}
	}

	// Replace a row by sum of itself and a
	for i := 0; i < order; i++ {
		for j := 0; j < order; j++ {
			if j != i {
				temp := augmented[j][i] / augmented[i][i]
				for k := 0; k < order*2; k++ {
					augmented[j][k] -= augmented[i][k] * temp
				}
			}
		}
	}

	for i := 0; i < order; i++ {
		temp := augmented[i][i]
		for j := 0; j < order*2; j++ {
			augmented[i][j] /= temp
		}
	}

	// building the result matrix
	result_data := make([][]float64, order)
	for i := range result_data {
		result_data[i] = make([]float64, order)
		for j := range result_data[i] {
			result_data[i][j] = augmented[i][order+j]
		}
	}
	return NewMatrix(result_data), nil

}

func (m *Matrix) String() string {
	builder := strings.Builder{}
	for _, line := range m.data {
		builder.WriteString("[")
		for _, val := range line {
			builder.WriteString(fmt.Sprintf("%v ", val))
		}
		builder.WriteString("]")
		builder.WriteString("\n")
	}
	return builder.String()
}
