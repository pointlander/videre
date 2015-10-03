package main

import (
	//"bufio"
	"flag"
	"fmt"
	"github.com/gonum/plot"
	"github.com/gonum/plot/plotter"
	"github.com/gonum/plot/vg"
	"github.com/gonum/plot/vg/draw"
	"github.com/pointlander/gonn/gonn"
	"github.com/skelterjohn/go.matrix"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	//"os"
	"regexp"
)

var (
	MASK   = [...]int{1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1}
	N      = len(MASK)
	OFFSET = N / 2
)

const (
	WIDTH = 1024
)

type Vector [WIDTH]float64

var cache = map[int64]*Vector{}

func randomVector(seed int64) *Vector {
	if vector, valid := cache[seed]; valid {
		return vector
	}
	vector, random := Vector{}, rand.New(rand.NewSource(seed))
	for i := range vector {
		//vector[i] = 2 * random.Float64() - 1
		vector[i] = random.Float64()
	}
	cache[seed] = &vector
	return &vector
}

func vectorForSymbol(symbol int) *Vector {
	return randomVector(int64(symbol))
}

func vectorForPosition(position int) *Vector {
	return randomVector(int64(position + 65536 + OFFSET))
}

func (a *Vector) add(b *Vector) *Vector {
	vector := Vector{}
	for i, j := range a {
		vector[i] = j + b[i]
	}
	return &vector
}

func (a *Vector) sub(b *Vector) *Vector {
	vector := Vector{}
	for i, j := range a {
		vector[i] = j - b[i]
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

/* https://en.wikipedia.org/wiki/Cosine_similarity */
func (a *Vector) similarity(b *Vector) float64 {
	cols := len(a)
	dot, maga, magb := float64(0), float64(0), float64(0)
	for j := 0; j < cols; j++ {
		aij, bij := a[j], b[j]
		dot += aij * bij
		maga += aij * aij
		magb += bij * bij
	}
	mag := math.Sqrt(maga * magb)
	if mag == 0 {
		return 0
	}
	return dot / mag
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
			x.Set(i, j, x.Get(i, j)/sum)
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
			m.Set(0, j, m.Get(0, j)+xij)
		}
		/* zero vectors skew the mean preventing good centering*/
		if !zero {
			n++
		}
	}
	for j := 0; j < cols; j++ {
		m.Set(0, j, m.Get(0, j)/float64(n))
	}
	return m
}

func subtract(x *matrix.DenseMatrix, y *matrix.DenseMatrix) {
	rows, cols := x.Rows(), x.Cols()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			x.Set(i, j, x.Get(i, j)-y.Get(0, j))
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
			z.Set(i, j, z.Get(i, j)/N)
		}
	}
	return z
}

type SVM struct {
	*matrix.DenseMatrix
	compress   *gonn.NeuralNetwork
	dictionary map[string]int
}

func NewSymbolVectorModel() SVM {
	return SVM{
		matrix.Zeros(256*256, WIDTH),
		gonn.DefaultNetwork(2*WIDTH, []int{WIDTH}, 2*WIDTH, false, 1),
		make(map[string]int),
	}
}

func (svm SVM) TrainFile(file string) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		log.Fatal(err)
	}
	for _, s := range regexp.MustCompile("\\s").Split(string(data), -1) {
		svm.dictionary[s]++
	}
	length := len(data) - OFFSET
	context := func(i int) int {
		var context int
		for j := i - 8; j < i; j++ {
			context = ((37 * context) + int(data[j])) & 0xFF
		}
		return (context << 8) | int(data[i])
	}
	for i := OFFSET + 8; i < length; i++ {
		c, o := &Vector{}, &Vector{}
		for j := -OFFSET; j <= OFFSET; j++ {
			if MASK[j+OFFSET] == 0 {
				continue
			}
			vector := vectorForSymbol(context(i + j))
			c = c.add(vector)
			o = o.add(vectorForPosition(j).mul(vector))
		}
		l := c.norm().add(o.norm())
		index := context(i)
		for j := 0; j < 1024; j++ {
			svm.Set(index, j, svm.Get(index, j)+l[j])
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

func (svm SVM) VectorFor(a string) *Vector {
	v, cols := Vector{}, svm.Cols()
	for b := 8; b < len(a); b++ {
		var i int
		for j := b - 8; j < b; j++ {
			i = ((37 * i) + int(a[j])) & 0xFF
		}
		i = (i << 8) | int(a[b])
		sum := float64(0)
		for j := 0; j < cols; j++ {
			x := svm.Get(i, j)
			sum += x * x
		}
		if sum == 0 {
			continue
		}
		sum = math.Sqrt(sum)
		for j := 0; j < cols; j++ {
			v[j] += svm.Get(i, j) / sum
		}
	}
	return &v
}

func (svm SVM) VectorForWord(a string) *Vector {
	v, cols := Vector{}, svm.Cols()
	b := len(a) - 1
	var i int
	for j := b - 8; j < b; j++ {
		i = ((37 * i) + int(a[j])) & 0xFF
	}
	i = (i << 8) | int(a[b])
	sum := float64(0)
	for j := 0; j < cols; j++ {
		x := svm.Get(i, j)
		sum += x * x
	}
	if sum == 0 {
		return &v
	}
	sum = math.Sqrt(sum)
	for j := 0; j < cols; j++ {
		v[j] += svm.Get(i, j) / sum
	}
	return &v
}

type SVMSet struct {
	svm SVM
	set [4096][2]int
}

func (s *SVMSet) Len() int {
	return len(s.set)
}

func (s *SVMSet) Fill(inputs, targets []float64, i int) {
	for j := 0; j < WIDTH; j++ {
		element := s.svm.Get(s.set[i][0], j)
		inputs[j], targets[j] = element, element
	}
	for j := 0; j < WIDTH; j++ {
		element := s.svm.Get(s.set[i][1], j)
		inputs[j+WIDTH], targets[j+WIDTH] = element, element
	}
}

func (svm SVM) TrainNN() {
	norm(svm.DenseMatrix)
	set := &SVMSet{svm: svm}
	for i := range set.set {
		set.set[i][0] = rand.Intn(256*256 - 1)
		set.set[i][1] = rand.Intn(256*256 - 1)
	}
	svm.compress.TrainSet(set, 1)
}

var (
	pca  = flag.Bool("pca", false, "train the model and then visualize with pca")
	hash = flag.Bool("hash", false, "train the model on tree books and then compare")
	word = flag.Bool("word", false, "word vector demo")
)

func main() {
	flag.Parse()

	if *word {
		svm := NewSymbolVectorModel()
		files, err := ioutil.ReadDir("data")
		if err != nil {
			log.Fatal(err)
		}
		for _, file := range files {
			fmt.Println("data/" + file.Name())
			svm.TrainFile("data/" + file.Name())
		}

		search := svm.VectorForWord(" the good king").norm()
		var near [100]struct {
			text       string
			similarity float64
		}
		for i := range near {
			near[i].similarity = -1
		}

		for text, count := range svm.dictionary {
			if count < 30 {
				continue
			}
			word := svm.VectorForWord(" the good " + text)
			similarity := search.similarity(word)
		INSERT:
			for i, n := range near {
				if similarity > n.similarity {
					near[i].text = text
					near[i].similarity = similarity
					i++
					for i < len(near) {
						tmp := near[i]
						near[i], n = n, tmp
						i++
					}
					break INSERT
				}
			}
		}
		fmt.Println(near)
	}

	if *hash {
		svm := NewSymbolVectorModel()
		svm.TrainFile("data/pg1661.txt")
		svm1 := NewSymbolVectorModel()
		svm1.TrainFile("data/pg2852.txt")
		s1 := svm1.Similarity(svm)
		fmt.Printf("similarity between two Doyle books: %v\n", s1)
		svm2 := NewSymbolVectorModel()
		svm2.TrainFile("data/pg2267.txt")
		s2 := svm2.Similarity(svm)
		fmt.Printf("similarity between Doyle and Shakespeare books: %v\n", s2)
	}

	if *pca {
		svm := NewSymbolVectorModel()
		svm.TrainFile("data/pg1661.txt")
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
