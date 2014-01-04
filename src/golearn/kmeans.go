package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

func Transpose(array [][]float64) [][]float64 {
	if len(array) == 0 {
		return array
	}
	transposed := make([][]float64, len(array[0]))
	for i, _ := range transposed {
		transposed[i] = make([]float64, len(array))
	}
	for i, _ := range array {
		for j, _ := range array[i] {
			transposed[j][i] = array[i][j]
		}
	}
	return transposed
}
func ArrayMin(array []float64) (min float64) {
	min = array[0]
	for _, v := range array[1:] {
		if v < min {
			min = v
		}
	}
	return
}
func ArrayMax(array []float64) (max float64) {
	max = array[0]
	for _, v := range array[1:] {
		if v > max {
			max = v
		}
	}
	return
}
func ArrayAverage(array []float64) (average float64) {
	average = 0
	for _, v := range array {
		average += v
	}
	average /= float64(len(array))
	return
}

func EuclideanDistance(v1, v2 []float64) (dist float64) {
	if len(v1) != len(v2) {
		panic("dim")
	}
	dist = 0.0
	for i, _ := range v1 {
		dist += math.Pow(v1[i]-v2[i], 2)
	}
	dist = math.Sqrt(dist)
	return
}

type Kmeans struct {
	data       [][]float64
	centroids  [][]float64
	assignment []int
}

func FitKmeans(data [][]float64, clusters int) *Kmeans {
	km := &Kmeans{}
	km.centroids = make([][]float64, clusters)
	km.data = data
	km.fit()
	return km
}

func (k *Kmeans) initialize_centroids() {
	t := Transpose(k.data)
	// get ranges for each dimension
	mins := make([]float64, 0, len(t))
	maxs := make([]float64, 0, len(t))
	for _, r := range t {
		mins = append(mins, ArrayMin(r))
		maxs = append(maxs, ArrayMax(r))
	}
	// generate randomized centroids
	for i, _ := range k.centroids {
		c := make([]float64, len(mins))
		for j, _ := range c {
			c[j] = rand.Float64()*(maxs[j]-mins[j]) + mins[j]
		}
		k.centroids[i] = c
	}
	k.generate_cluster_assignments()
}

func (k *Kmeans) generate_cluster_assignments() {
	// get datapoint-cluster assignment
	assignment := make([]int, len(k.data))
	for i, dv := range k.data {
		assignment[i] = 0
		min_dist := EuclideanDistance(dv, k.centroids[0])
		for j, cv := range k.centroids[1:] {
			dist := EuclideanDistance(dv, cv)
			if dist < min_dist {
				assignment[i] = j
				min_dist = dist
			}
		}
	}
	k.assignment = assignment
	return
}

func (k *Kmeans) recalculate_centroids() (changed bool, err string) {
	// calculate new centroids as average of all assigned datapoints
	for i, centroid := range k.centroids {
		// retrieve assigned points
		points := make([][]float64, 0)
		for j, cluster := range k.assignment {
			if i == cluster {
				points = append(points, k.data[j])
			}
		}
		// calculate mean of all points
		if len(points) == 0 {
			continue
		}
		points = Transpose(points)
		for j, _ := range centroid {
			new_cent := ArrayAverage(points[j])
			if new_cent != centroid[j] {
				centroid[j] = new_cent
				changed = true
			}
		}
	}
	k.generate_cluster_assignments()
	return
}

func (k *Kmeans) fit() {
	rand.Seed(time.Now().UnixNano())
	k.initialize_centroids()
	changed, _ := k.recalculate_centroids()
	for {
		changed, _ = k.recalculate_centroids()
		if !changed {
			break
		}
	}
}

func main() {
	z := [][]float64{
		{0.0, 3.0, 1.0, 3.3},
		{1.0, 2.0, 10.0, 31.3},
		{3.0, 4.0, 5.0, 23.1},
		{1.0, 3.0, 3.0, 54.1},
		{1.0, 0.0, 2.0, 23.3},
		{1.0, 9.0, 33.0, 32.2},
	}

	k := FitKmeans(z, 2)
	fmt.Printf("\ndone\ncentroids: %v\ndata:%v\n", k.centroids, k.data)
}
