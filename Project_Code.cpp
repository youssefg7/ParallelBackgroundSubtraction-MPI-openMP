#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <sys/stat.h>
#include <string>
#include <mpi.h>
#include <filesystem>
namespace fs = std::filesystem;

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IN_PICS 500 // number of input images to use for background calculation
#define IN_CHANNELS 3 // input is colored images
#define OUT_CHANNELS 1 // output is a binary mask 
#define THRESHOLD 30.0 // threshold per channel difference between the pixel values of new frame and background
int N_THREADS = 2; // number of threads for the outer loop in OpenMP
#define SOURCE_IMGS_DIR "/Volumes/D/Uni/Semester 10/CSE445 High Performance Computing/Project/BackgroundSubtraction/GroundtruthSeq/input"
#define TEST_IMG_PATH "test.jpg"


void mpi_parallel_partial_bg_calc(std::vector<unsigned char*> images, int width, int height, int rank, int size) {

    // Calculate the chunk size and start and end indices for the current rank
    int chunk = height / (size - 1);
    int start = (rank - 1) * chunk;
    int end = rank * chunk;
    if (rank == size - 1) {
        end = height;
    }

    // Allocate memory for the process portion of the background image
    unsigned char* backgroundPortion = new unsigned char[height * width * IN_CHANNELS];

    for (int y = start; y < end; ++y) {
        for (int x = 0; x < width; ++x) {
            long long sum[3] = { 0, 0, 0 };
            for (auto img : images) {
                int idx = (y * width + x) * IN_CHANNELS;
                sum[0] += img[idx];
                sum[1] += img[idx + 1];
                sum[2] += img[idx + 2];
            }

            backgroundPortion[(y * width + x) * IN_CHANNELS] = sum[0] / images.size();
            backgroundPortion[(y * width + x) * IN_CHANNELS + 1] = sum[1] / images.size();
            backgroundPortion[(y * width + x) * IN_CHANNELS + 2] = sum[2] / images.size();
        }
    }
    // Send the portion of the background image back to rank 0 master process
    MPI_Send(backgroundPortion, width * height * IN_CHANNELS, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
}

void mpi_parallel_mask_calc(unsigned char* background, unsigned char* newFrame, int width, int height, int rank, int size) {

    // Calculate the chunk size and start and end indices for the current rank
    int chunk = height / (size - 1);
    int start = (rank - 1) * chunk;
    int end = rank * chunk;
    if (rank == size - 1) {
        end = height;
    }

    // Allocate memory for the process portion of the foreground mask
    unsigned char* foregroundMaskPortion = new unsigned char[width * height * 1];

    for (int y = start; y < end; ++y) {
        for (int x = 0; x < width; ++x) {
            int bg_idx = (y * width + x) * IN_CHANNELS;
            int fg_idx = y * width + x;
            int sm = 0;
            for (int c = 0; c < IN_CHANNELS; ++c) {
                sm += std::abs(background[bg_idx + c] - newFrame[bg_idx + c]);
            }
            foregroundMaskPortion[fg_idx] = sm > (THRESHOLD * IN_CHANNELS) ? 255 : 0;
        }
    }
    // Send the portion of the foreground mask back to rank 0 master process
    MPI_Send(foregroundMaskPortion, width * height * OUT_CHANNELS, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
}

unsigned char* omp_parallel_bg_calc(std::vector<unsigned char*> images, int width, int height) {
    // Allocate memory for the background image
    unsigned char* background = new unsigned char[width * height * IN_CHANNELS];

#pragma omp parallel for num_threads(N_THREADS)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            long long sum[3] = { 0, 0, 0 };
            for (auto img : images) {
                int idx = (y * width + x) * IN_CHANNELS;
                sum[0] += img[idx];
                sum[1] += img[idx + 1];
                sum[2] += img[idx + 2];
            }
            background[(y * width + x) * IN_CHANNELS] = sum[0] / images.size();
            background[(y * width + x) * IN_CHANNELS + 1] = sum[1] / images.size();
            background[(y * width + x) * IN_CHANNELS + 2] = sum[2] / images.size();
        }
    }
    return background;
}

unsigned char* omp_parallel_mask_calc(unsigned char* background, unsigned char* newFrame, int width, int height) {
    // Allocate memory for the foreground mask
    unsigned char* foregroundMask = new unsigned char[width * height * OUT_CHANNELS];

#pragma omp parallel for num_threads(N_THREADS)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int bg_idx = (y * width + x) * IN_CHANNELS;
            int fg_idx = y * width + x;
            int sm = 0;
            for (int c = 0; c < IN_CHANNELS; ++c) {
                sm += std::abs(background[bg_idx + c] - newFrame[bg_idx + c]);
            }
            foregroundMask[fg_idx] = sm > (THRESHOLD * IN_CHANNELS) ? 255 : 0;
        }
    }
    return foregroundMask;
}

unsigned char* serial_bg_calc(std::vector<unsigned char*> images, int width, int height) {
    // Allocate memory for the background image
    unsigned char* background = new unsigned char[width * height * IN_CHANNELS];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            long long sum[3] = { 0, 0, 0 };
            for (auto img : images) {
                int idx = (y * width + x) * IN_CHANNELS;
                sum[0] += img[idx];
                sum[1] += img[idx + 1];
                sum[2] += img[idx + 2];
            }
            background[(y * width + x) * IN_CHANNELS] = sum[0] / images.size();
            background[(y * width + x) * IN_CHANNELS + 1] = sum[1] / images.size();
            background[(y * width + x) * IN_CHANNELS + 2] = sum[2] / images.size();
        }
    }
    return background;
}

unsigned char* serial_mask_calc(unsigned char* background, unsigned char* newFrame, int width, int height) {
    // Allocate memory for the foreground mask
    unsigned char* foregroundMask = new unsigned char[width * height * OUT_CHANNELS];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int bg_idx = (y * width + x) * IN_CHANNELS;
            int fg_idx = y * width + x;
            int sm = 0;
            for (int c = 0; c < IN_CHANNELS; ++c) {
                sm += std::abs(background[bg_idx + c] - newFrame[bg_idx + c]);
            }
            foregroundMask[fg_idx] = sm > (THRESHOLD * IN_CHANNELS) ? 255 : 0;
        }
    }
    return foregroundMask;
}


int main(int argc, char** argv) {
    // Initialize MPI and get the rank and size of the communicator
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    N_THREADS = size;
    //////////////////////////////////////////////// Load the images and test frame ////////////////////////////////////////////////

    // Load the images using stb_image library
    std::string folderPath = SOURCE_IMGS_DIR;
    std::vector<unsigned char*> images;
    int count = 0;
    int width, height, channels;
    int k = 0;
    // Load the images
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.path().extension() == ".jpg") {
            unsigned char* img = stbi_load(entry.path().string().c_str(), &width, &height, &channels, 0);
            images.push_back(img);
            count++;
        }
        if (count == IN_PICS) break;
    }

    unsigned char* testFrame;
    testFrame = stbi_load(TEST_IMG_PATH, &width, &height, &channels, 0);

    if (rank == 0)
    {
        printf("Images loaded: %d\n\n", (int)images.size());
        //////////////////////////////////////////////////////////////// Serial ////////////////////////////////////////////////////////////////
        // Compute the background in serial 
        printf("Running Serial...\n");
        auto start = std::chrono::high_resolution_clock::now();

        unsigned char* background_serial = serial_bg_calc(images, width, height);

        auto done = std::chrono::high_resolution_clock::now();
        auto taken_time = std::chrono::duration_cast<std::chrono::microseconds>(done - start).count();
        printf("Time taken using serial to calculate background: %lld microseconds.\n", taken_time);

        // Compute the foreground mask in serial
        start = std::chrono::high_resolution_clock::now();

        unsigned char* foregroundMask_serial = serial_mask_calc(background_serial, testFrame, width, height);

        done = std::chrono::high_resolution_clock::now();
        taken_time = std::chrono::duration_cast<std::chrono::microseconds>(done - start).count();
        printf("Time taken using serial to calculate foreground mask: %lld microseconds.\n", taken_time);

        // Save the background and foreground mask
        stbi_write_jpg("serial_background.jpg", width, height, IN_CHANNELS, background_serial, 100);
        stbi_write_jpg("serial_foregroundMask.jpg", width, height, OUT_CHANNELS, foregroundMask_serial, 100);



        //////////////////////////////////////////////////////////////// OpenMP ////////////////////////////////////////////////////////////////
        // Enable nested parallelism for OpenMP
        
        printf("\nRunning OpenMP with %d threads...\n", N_THREADS);
        // Compute the background in parallel using OpenMP
        start = std::chrono::high_resolution_clock::now();

        unsigned char* background_omp = omp_parallel_bg_calc(images, width, height);

        done = std::chrono::high_resolution_clock::now();
        taken_time = std::chrono::duration_cast<std::chrono::microseconds>(done - start).count();
        printf("Time taken using openMP to calculate background: %lld microseconds.\n", taken_time);

        // Compute the foreground mask in paral lel using OpenMP
        start = std::chrono::high_resolution_clock::now();
        unsigned char* foregroundMask = omp_parallel_mask_calc(background_omp, testFrame, width, height);
        done = std::chrono::high_resolution_clock::now();
        taken_time = std::chrono::duration_cast<std::chrono::microseconds>(done - start).count();
        printf("Time taken using openMP to calculate foreground mask: %lld microseconds.\n", taken_time);

        // Save the background and foreground mask
        stbi_write_jpg("omp_background.jpg", width, height, IN_CHANNELS, background_omp, 100);
        stbi_write_jpg("omp_foregroundMask.jpg", width, height, OUT_CHANNELS, foregroundMask, 100);


        //////////////////////////////////////////////////////////////// MPI ////////////////////////////////////////////////////////////////
        
        printf("\nRunning MPI with %d processes...\n", size);
        
        int img_size = width * height * IN_CHANNELS;

        // allocate memory for the background image
        unsigned char* background_mpi = new unsigned char[img_size];

        // Compute the background in parallel using MPI
        start = std::chrono::high_resolution_clock::now();

        // Send a signal to other ranks to start the background calculation
        for (int i = 1; i < size; i++)
        {
            MPI_Send(&i, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        // Collect the background image chunks, only rank 0 will save the images when all ranks are done
        int chunk = height / (size - 1);
        int s, end;
        for (int i = 1; i < size; i++) {
            unsigned char* backgroundPortion = new unsigned char[img_size];
            MPI_Recv(backgroundPortion, img_size, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            s = (i - 1) * chunk;
            end = i * chunk;
            if (i == size - 1) {
                end = height;
            }
            for (int y = s; y < end; y++) {
                for (int x = 0; x < width; x++) {
                    for (int c = 0; c < IN_CHANNELS; c++) {
                        background_mpi[(y * width + x) * IN_CHANNELS + c] = backgroundPortion[(y * width + x) * IN_CHANNELS + c];
                    }
                }
            }
        }

        done = std::chrono::high_resolution_clock::now();
        taken_time = std::chrono::duration_cast<std::chrono::microseconds>(done - start).count();
        printf("Time taken using MPI to calculate background: %lld microseconds.\n", taken_time);

        // Save the background image
        stbi_write_jpg("mpi_background.jpg", width, height, IN_CHANNELS, background_mpi, 100);


        // Compute the foreground mask in parallel using MPI
        start = std::chrono::high_resolution_clock::now();

        // Send background and test frame to other ranks
        for (int i = 1; i < size; i++)
        {
            MPI_Send(background_mpi, img_size, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);
        }


        // allocate memory for the foreground mask
        unsigned char* foregroundMask_mpi = new unsigned char[width * height * 1];

        // Collect the foreground mask chunks, only rank 0 will save the images when all ranks are done
        for (int i = 1; i < size; i++) {
            unsigned char* foregroundMaskPortion = new unsigned char[height * width * 1];
            MPI_Recv(foregroundMaskPortion, height * width * 1, MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            s = (i - 1) * chunk;
            end = i * chunk;
            if (i == size - 1) {
                end = height;
            }
            for (int y = s; y < end; y++) {
                for (int x = 0; x < width; x++) {
                    foregroundMask_mpi[y * width + x] = foregroundMaskPortion[y * width + x];
                }
            }
        }

        done = std::chrono::high_resolution_clock::now();
        taken_time = std::chrono::duration_cast<std::chrono::microseconds>(done - start).count();
        printf("Time taken using MPI to calculate foreground mask: %lld microseconds.\n", taken_time);

        // Save the foreground mask
        stbi_write_jpg("mpi_foregroundMask.jpg", width, height, OUT_CHANNELS, foregroundMask_mpi, 100);

        // Cleanup the memory 
        for (auto img : images) {
            stbi_image_free(img);
        }
        stbi_image_free(testFrame);

        delete[] background_serial;
        delete[] foregroundMask_serial;
        delete[] background_omp;
        delete[] foregroundMask;
        delete[] background_mpi;
        delete[] foregroundMask_mpi;

    }
    else {
        // Blocked until rank 0 sends a signal to start the background calculation
        int signal;
        MPI_Recv(&signal, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (signal == rank)
        {
            int img_size = width * height * IN_CHANNELS;

            // calculate the background portion and send it back to rank 0
            mpi_parallel_partial_bg_calc(images, width, height, rank, size);

            // Receive background and test frame from rank 0
            unsigned char* background = new unsigned char[img_size];
            MPI_Recv(background, img_size, MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // calculate the foreground mask portion and send it back to rank 0
            mpi_parallel_mask_calc(background, testFrame, width, height, rank, size);
        }
    }
    // Finalize MPI
    MPI_Finalize();
    return 0;
}
