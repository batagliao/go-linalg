package linalg

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_New(t *testing.T) {
	data := [][]float64{
		{1, 2, 3, 4},
		{4, 3, 2, 1},
	}
	m := NewMatrix(data)

	assert.Equal(t, 2, m.Lines())
	assert.Equal(t, 4, m.Columns())

	data = [][]float64{
		{1, 2, 3, 4},
	}
	m = NewMatrix(data)
	assert.Equal(t, 1, m.Lines())
	assert.Equal(t, 4, m.Columns())

	data = [][]float64{
		{},
	}
	m = NewMatrix(data)
	assert.Equal(t, 0, m.Lines())
	assert.Equal(t, 0, m.Columns())

	data = [][]float64{}
	m = NewMatrix(data)
	assert.Equal(t, 0, m.Lines())
	assert.Equal(t, 0, m.Columns())
}

func Test_NullMatrix(t *testing.T) {
	null := NullMatrix(5)
	assert.Equal(t, 5, null.Lines())
	assert.Equal(t, 5, null.Columns())

	val, err := null.Position(3, 3)
	assert.Nil(t, err)
	assert.Equal(t, 0., val)
}

func Test_Position(t *testing.T) {
	null := NullMatrix(2)
	_, err := null.Position(3, 1)
	assert.Error(t, err)

	_, err2 := null.Position(1, 3)
	assert.Error(t, err2)

	data := [][]float64{
		{1, 2, 3, 4},
		{4, 3, 2, 1},
	}
	m := NewMatrix(data)
	val, err := m.Position(1, 3)
	assert.Nil(t, err)
	assert.Equal(t, 3., val)
}

func Test_NewIdentityMatrix(t *testing.T) {
	m := NewIdentityMatrix(5)

	assert.Equal(t, 5, m.Lines())
	assert.Equal(t, 5, m.Columns())

	pos, err := m.Position(1, 1)
	assert.Nil(t, err)
	assert.Equal(t, 1., pos)

	pos, err = m.Position(2, 2)
	assert.Nil(t, err)
	assert.Equal(t, 1., pos)

	pos, err = m.Position(3, 3)
	assert.Nil(t, err)
	assert.Equal(t, 1., pos)

	pos, err = m.Position(4, 4)
	assert.Nil(t, err)
	assert.Equal(t, 1., pos)

	pos, err = m.Position(5, 5)
	assert.Nil(t, err)
	assert.Equal(t, 1., pos)

	pos, err = m.Position(5, 1)
	assert.Nil(t, err)
	assert.Equal(t, 0., pos)

	pos, err = m.Position(2, 3)
	assert.Nil(t, err)
	assert.Equal(t, 0., pos)

	_, err = m.Position(0, 3)
	assert.Error(t, err)
}

func Test_Sum(t *testing.T) {
	A := NewMatrix([][]float64{
		{1, 2, 3, 4},
		{4, 3, 2, 1},
	})

	B := NewMatrix([][]float64{
		{-5, 8, 1, 3},
		{0, 2, 4, -9},
	})

	C := NewMatrix([][]float64{
		{1, 2, 3, 4},
	})

	_, err := A.Sum(C)
	assert.Error(t, err)

	_, err = C.Sum(B)
	assert.Error(t, err)

	result, _ := A.Sum(B)

	assert.Equal(t, A.Lines(), result.Lines())
	assert.Equal(t, A.Columns(), result.Columns())

	pos, err := result.Position(1, 1)
	assert.Nil(t, err)
	assert.Equal(t, -4., pos)

	pos, _ = result.Position(1, 2)
	assert.Equal(t, 10., pos)

	pos, _ = result.Position(1, 3)
	assert.Equal(t, 4., pos)

	pos, _ = result.Position(1, 4)
	assert.Equal(t, 7., pos)

	pos, _ = result.Position(2, 1)
	assert.Equal(t, 4., pos)

	pos, _ = result.Position(2, 2)
	assert.Equal(t, 5., pos)

	pos, _ = result.Position(2, 3)
	assert.Equal(t, 6., pos)

	pos, _ = result.Position(2, 4)
	assert.Equal(t, -8., pos)
}

func Test_Sub(t *testing.T) {
	A := NewMatrix([][]float64{
		{1, 2, 3, 4},
		{4, 3, 2, 1},
	})

	B := NewMatrix([][]float64{
		{-5, 8, 1, 3},
		{0, 2, 4, -9},
	})

	C := NewMatrix([][]float64{
		{1, 2, 3, 4},
	})

	_, err := A.Sub(C)
	assert.Error(t, err)

	_, err = C.Sub(B)
	assert.Error(t, err)

	result, _ := A.Sub(B)

	assert.Equal(t, A.Lines(), result.Lines())
	assert.Equal(t, A.Columns(), result.Columns())

	pos, err := result.Position(1, 1)
	assert.Nil(t, err)
	assert.Equal(t, 6., pos)

	pos, _ = result.Position(1, 2)
	assert.Equal(t, -6., pos)

	pos, _ = result.Position(1, 3)
	assert.Equal(t, 2., pos)

	pos, _ = result.Position(1, 4)
	assert.Equal(t, 1., pos)

	pos, _ = result.Position(2, 1)
	assert.Equal(t, 4., pos)

	pos, _ = result.Position(2, 2)
	assert.Equal(t, 1., pos)

	pos, _ = result.Position(2, 3)
	assert.Equal(t, -2., pos)

	pos, _ = result.Position(2, 4)
	assert.Equal(t, 10., pos)
}

func Test_ScalarProduct(t *testing.T) {
	A := NewMatrix([][]float64{
		{1, 2, 3, 4},
		{4, 3, 2, 1},
	})

	result := A.ScalarProduct(2)

	assert.Equal(t, A.Lines(), result.Lines())
	assert.Equal(t, A.Columns(), result.Columns())

	pos, err := result.Position(1, 1)
	assert.Nil(t, err)
	assert.Equal(t, 2., pos)

	pos, _ = result.Position(1, 2)
	assert.Equal(t, 4., pos)

	pos, _ = result.Position(1, 3)
	assert.Equal(t, 6., pos)

	pos, _ = result.Position(1, 4)
	assert.Equal(t, 8., pos)

	pos, _ = result.Position(2, 1)
	assert.Equal(t, 8., pos)

	pos, _ = result.Position(2, 2)
	assert.Equal(t, 6., pos)

	pos, _ = result.Position(2, 3)
	assert.Equal(t, 4., pos)

	pos, _ = result.Position(2, 4)
	assert.Equal(t, 2., pos)

	result = A.ScalarProduct(0.5)

	assert.Equal(t, A.Lines(), result.Lines())
	assert.Equal(t, A.Columns(), result.Columns())

	pos, err = result.Position(1, 1)
	assert.Nil(t, err)
	assert.Equal(t, 0.5, pos)

	pos, _ = result.Position(1, 2)
	assert.Equal(t, 1., pos)

	pos, _ = result.Position(1, 3)
	assert.Equal(t, 1.5, pos)

	pos, _ = result.Position(1, 4)
	assert.Equal(t, 2., pos)

	pos, _ = result.Position(2, 1)
	assert.Equal(t, 2., pos)

	pos, _ = result.Position(2, 2)
	assert.Equal(t, 1.5, pos)

	pos, _ = result.Position(2, 3)
	assert.Equal(t, 1., pos)

	pos, _ = result.Position(2, 4)
	assert.Equal(t, 0.5, pos)

}

func Test_Transpose(t *testing.T) {
	data := [][]float64{
		{1, 2, 3, 4},
		{4, 3, 2, 1},
	}
	A := NewMatrix(data)

	expected := A.Transpose()
	assert.Equal(t, expected.Lines(), A.Columns())
	assert.Equal(t, expected.Columns(), A.Lines())

	pos, _ := expected.Position(1, 1)
	assert.Equal(t, 1., pos)

	pos, _ = expected.Position(1, 2)
	assert.Equal(t, 4., pos)

	pos, _ = expected.Position(2, 1)
	assert.Equal(t, 2., pos)

	pos, _ = expected.Position(2, 2)
	assert.Equal(t, 3., pos)

	pos, _ = expected.Position(3, 1)
	assert.Equal(t, 3., pos)

	pos, _ = expected.Position(3, 2)
	assert.Equal(t, 2., pos)

	pos, _ = expected.Position(4, 1)
	assert.Equal(t, 4., pos)

	pos, _ = expected.Position(4, 2)
	assert.Equal(t, 1., pos)
}

func Test_Product(t *testing.T) {
	A := NewMatrix([][]float64{
		{-2, 8, 1},
		{3, 1, 6},
	})

	B := NewMatrix([][]float64{
		{1, 2},
		{-4, 3},
		{-2, 5},
	})

	result, err := A.Product(B)
	assert.Nil(t, err)
	assert.Equal(t, 2, result.Lines())
	assert.Equal(t, 2, result.Columns())

	pos, _ := result.Position(1, 1)
	assert.Equal(t, -36., pos)

	pos, _ = result.Position(1, 2)
	assert.Equal(t, 25., pos)

	pos, _ = result.Position(2, 1)
	assert.Equal(t, -13., pos)

	pos, _ = result.Position(2, 2)
	assert.Equal(t, 39., pos)

	C := NewMatrix([][]float64{
		{1, 2, 3, 4},
	})

	_, err = A.Product(C)
	assert.Error(t, err)
}

func Test_LU(t *testing.T) {
	A := NewMatrix([][]float64{
		{2, -1, -2},
		{-4, 6, 3},
		{-4, -2, 8},
	})

	L := NewMatrix([][]float64{
		{1, 0, 0},
		{-2, 1, 0},
		{-2, -1, 1},
	})

	U := NewMatrix([][]float64{
		{2, -1, -2},
		{0, 4, -1},
		{0, 0, 3},
	})

	l, u, err := A.LU()
	assert.Nil(t, err)
	assert.True(t, L.Equals(l))
	assert.True(t, U.Equals(u))

	// asserting property
	// if B = L * U
	// then A and B should be equal
	// This proves the decomposition preserves the original matrix values
	B, err := L.Product(U)
	assert.Nil(t, err)
	assert.True(t, A.Equals(B))

}

func Test_Determinant(t *testing.T) {
	empty := empty_matrix
	det, err := empty.Determinant()
	assert.Error(t, err)
	assert.Equal(t, 0., det)

	data := [][]float64{
		{3},
	}

	A := NewMatrix(data)
	det, err = A.Determinant()
	assert.Nil(t, err)
	assert.Equal(t, 3., det)

	data = [][]float64{
		{0, 2},
		{1, -1},
	}
	B := NewMatrix(data)
	det, err = B.Determinant()
	assert.Nil(t, err)
	assert.Equal(t, -2., det)

	C := NewMatrix([][]float64{
		{1, 3, 10},
		{-1, 1, 10},
		{0, 2, 10},
	})
	det, _ = C.Determinant()
	assert.Equal(t, 0., det)
}

func Test_Equals(t *testing.T) {
	A := NewMatrix([][]float64{
		{1, 2, 3, 4},
		{4, 3, 2, 1},
	})

	B := NewMatrix([][]float64{
		{1, 2, 3, 4},
		{4, 3, 2, 1},
	})

	C := NewMatrix([][]float64{
		{1, 2, 3, 4},
		{4, 1, 3, 2},
	})

	D := NewMatrix([][]float64{
		{1, 2, 3, 4},
	})

	assert.True(t, A.Equals(B))
	assert.True(t, B.Equals(A))
	assert.False(t, A.Equals(C))
	assert.False(t, B.Equals(D))
}

func Test_Inverse(t *testing.T) {
	A := NewMatrix([][]float64{
		{5, 7, 9},
		{4, 3, 8},
		{7, 5, 6},
	})

	Ai_expected := NewMatrix([][]float64{
		{-0.20952380952380953, 0.028571428571428605, 0.2761904761904762},
		{0.3047619047619048, -0.3142857142857144, -0.0380952380952381},
		{-0.009523809523809552, 0.22857142857142862, -0.1238095238095238},
	})

	inv, err := A.Inverse()
	assert.Nil(t, err)
	assert.True(t, Ai_expected.Equals(inv))

	// inverse of the identity is equals to the identity
	I := NewIdentityMatrix(2)
	Ii, _ := I.Inverse()
	assert.True(t, I.Equals(Ii))
}
