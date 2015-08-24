package main

import (
	"flag"
	"fmt"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg"
	"github.com/gonum/plot/vg/draw"
	"github.com/skelterjohn/go.matrix"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
)

const (
	N = 5
	OFFSET = N/2
)

type Vector [1024]float64

var cache = map[int64]*Vector{}

func randomVector(seed int64) *Vector {
	if vector, valid := cache[seed]; valid {
		return vector
	}
	vector, random := Vector{}, rand.New(rand.NewSource(seed))
	for i := range vector {
		vector[i] = 2 * random.Float64() - 1
	}
	cache[seed] = &vector
	return &vector
}

func vectorForSymbol(symbol byte) *Vector {
	return randomVector(int64(symbol))
}

func vectorForPosition(position int) *Vector {
	return randomVector(int64(position + 256 + OFFSET))
}

func (a *Vector) add(b *Vector) *Vector {
	vector := Vector{}
	for i, j := range a {
		vector[i] = j + b[i]
	}
	return &vector
}

func (a *Vector) mul(b *Vector) *Vector {
	vector := Vector{}
	for i, j := range a {
		vector[i] = j * b[i]
	}
	return &vector
}

func (a *Vector) norm() *Vector {
	var sum float64
	for _, j := range a {
		sum += j * j
	}
	sum = math.Sqrt(sum)
	if sum == 0 {
		return a
	}
	vector := Vector{}
	for i, j := range a {
		vector[i] = j / sum
	}
	return &vector
}

func norm(x *matrix.DenseMatrix) {
	rows, cols := x.Rows(), x.Cols()
	for i := 0; i < rows; i++ {
		sum := float64(0)
		for j := 0; j < cols; j++ {
			k := x.Get(i, j)
			sum += k * k
		}
		sum = math.Sqrt(sum)
		if sum == 0 {
			continue
		}
		for j := 0; j < cols; j++ {
			x.Set(i, j, x.Get(i, j) / sum)
		}
	}
}

func mean(x *matrix.DenseMatrix) *matrix.DenseMatrix {
	rows, cols := x.Rows(), x.Cols()
	m, n := matrix.Zeros(1, cols), 0
	for i := 0; i < rows; i++ {
		zero := true
		for j := 0; j < cols; j++ {
			xij := x.Get(i, j)
			if xij != 0 {
				zero = false
			}
			m.Set(0, j, m.Get(0, j) + xij)
		}
		/* zero vectors skew the mean preventing good centering*/
		if !zero {
			n++
		}
	}
	for j := 0; j < cols; j++ {
		m.Set(0, j, m.Get(0, j) / float64(n))
	}
	return m
}

func subtract(x *matrix.DenseMatrix, y *matrix.DenseMatrix) {
	rows, cols := x.Rows(), x.Cols()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			x.Set(i, j, x.Get(i, j) - y.Get(0, j))
		}
	}
}

func cov(x *matrix.DenseMatrix) *matrix.DenseMatrix {
	y := x.Transpose()
	z, err := y.TimesDense(x)
	if err != nil {
		log.Fatal(err)
	}
	N, rows, cols := float64(x.Rows()), z.Rows(), z.Cols()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			z.Set(i, j, z.Get(i, j) / N)
		}
	}
	return z
}

type SVM struct {
	*matrix.DenseMatrix
}

func NewSymbolVectorModel() SVM {
	return SVM{matrix.Zeros(256 * 256, 1024)}
}

func (svm SVM) TrainFile(file string) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		log.Fatal(err)
	}
	data = append(make([]byte, OFFSET), data...)
	length := len(data) - OFFSET
	for i := OFFSET; i < length; i++ {
		c := &Vector{}
		for j := -OFFSET; j <= OFFSET; j++ {
			if j == 0 {
				continue
			}
			c = c.add(vectorForSymbol(data[i + j]))
		}
		o := &Vector{}
		for j := -OFFSET; j <= OFFSET; j++ {
			if j == 0 {
				continue
			}
			o = o.add(vectorForPosition(j).mul(vectorForSymbol(data[i + j])))
		}
		l := c.norm().add(o.norm())
		for j := 0; j < 1024; j++ {
			index := (int(data[i - 1]) << 8) | int(data[i])
			svm.Set(index, j, svm.Get(index, j) + l[j])
		}
	}
}

/* http://nghiaho.com/?page_id=1030 */
func (svm SVM) PCA() *matrix.DenseMatrix {
	m := svm.DenseMatrix
	norm(m)
	subtract(m, mean(m))
	var v *matrix.DenseMatrix
	{
		_, _, V, err := cov(m).SVD()
		if err != nil {
			log.Fatal(err)
		}
		rows := V.Rows()
		v = matrix.Zeros(rows, 2)
		for i := 0; i < rows; i++ {
			for j := 0; j < 2; j++ {
				v.Set(i, j, V.Get(i, j))
			}
		}
	}
	r, err := m.TimesDense(v)
	if err != nil {
		log.Fatal(err)
	}
	return r
}

/* https://en.wikipedia.org/wiki/Cosine_similarity */
func (a SVM) Similarity(b SVM) float64 {
	var s float64
	rows, cols := a.Rows(), a.Cols()
	n := 0
	for i := 0; i < rows; i++ {
		dot, maga, magb := float64(0), float64(0), float64(0)
		for j := 0; j < cols; j++ {
			aij, bij := a.Get(i, j), b.Get(i, j)
			dot += aij * bij
			maga += aij * aij
			magb += bij * bij
		}
		mag := math.Sqrt(maga * magb)
		if mag == 0 {
			continue
		}
		s += dot / mag
		n++
	}
	return s / float64(n)
}

var (
	pca = flag.Bool("pca", false, "train the model and then visualize with pca")
	hash = flag.Bool("hash", false, "train the model on tree books and then compare")
)

func main() {
	flag.Parse()

	svm := NewSymbolVectorModel()
	svm.TrainFile("pg1661.txt")

	if *hash {
		svm1 := NewSymbolVectorModel()
		svm1.TrainFile("pg2852.txt")
		s1 := svm1.Similarity(svm)
		fmt.Printf("similarity between two Doyle books: %v\n", s1)
		svm2 := NewSymbolVectorModel()
		svm2.TrainFile("pg2267.txt")
		s2 := svm2.Similarity(svm)
		fmt.Printf("similarity between Doyle and Shakespeare books: %v\n", s2)
	}

	if *pca {
		r := svm.PCA()
		rows := r.Rows()
		points := make(plotter.XYs, rows)
		for i := 0; i < rows; i++ {
			points[i].X, points[i].Y = r.Get(i, 0), r.Get(i, 1)
		}

		p, err := plot.New()
		if err != nil {
			log.Fatal(err)
		}
		p.Title.Text = "Symbol Vectors"
		p.X.Label.Text = "X"
		p.Y.Label.Text = "Y"
		scatter, err := plotter.NewScatter(points)
		if err != nil {
			log.Fatal(err)
		}
		scatter.Shape = draw.CircleGlyph{}
		scatter.Radius = vg.Points(1)
		p.Add(scatter)
		if err := p.Save(512, 512, "symbol_vectors.png"); err != nil {
			log.Fatal(err)
		}
	}
}
