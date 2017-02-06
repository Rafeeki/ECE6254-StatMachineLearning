// Demonstrate simple MPI program
// George F. Riley, Georgia Tech, Fall 2011


#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "mpi.h"

using namespace std;

#define MSG_SIZE 1000
char buf[MSG_SIZE];  // Message contents

int main(int argc,char**argv)
{
  int  numtasks, rank, rc; 
  
  rc = MPI_Init(&argc,&argv);
  if (rc != MPI_SUCCESS) {
    printf ("Error starting MPI program. Terminating.\n");
    MPI_Abort(MPI_COMM_WORLD, rc);
  }

  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  printf ("Number of tasks= %d My rank= %d\n", numtasks,rank);
  for (int round = 0; round < 100;++round)
    {
      if (rank == 0)
        { // Rank zero sends first then receives, all other 
          // receive first then send
          cout << "Rank " << rank
               << " sending to rank" << rank + 1
               << " round " << round << endl;
          rc = MPI_Send(buf, sizeof(buf), MPI_CHAR, rank + 1,
                        0, MPI_COMM_WORLD);
          if (rc != MPI_SUCCESS)
            {
              cout << "Rank " << rank
                   << " send failed, rc " << rc << endl;
              MPI_Finalize();
              exit(1);
            }
          MPI_Status status;
          rc = MPI_Recv(buf, sizeof(buf), MPI_CHAR, MPI_ANY_SOURCE,
                        0, MPI_COMM_WORLD, &status);
          if (rc != MPI_SUCCESS)
            {
              cout << "Rank " << rank
                   << " recv failed, rc " << rc << endl;
              MPI_Finalize();
              exit(1);
            }
          int count = 0;
          MPI_Get_count(&status, MPI_CHAR, &count);
          cout << "Rank " << rank 
               << " received " << count << " bytes from "
               << status.MPI_SOURCE << endl;
        }
      else
        {
          MPI_Status status;
          rc = MPI_Recv(buf, sizeof(buf), MPI_CHAR, MPI_ANY_SOURCE,
                        0, MPI_COMM_WORLD, &status);
          if (rc != MPI_SUCCESS)
            {
              cout << "Rank " << rank
                   << " recv failed, rc " << rc << endl;
              MPI_Finalize();
              exit(1);
            }
          // Now send to next rank (0 if we are last rank)
          int nextRank = rank + 1;
          if (nextRank == numtasks) nextRank = 0;
          cout << "Rank " << rank
               << " sending to rank" << nextRank
               << " round " << round << endl;
          rc = MPI_Send(buf, sizeof(buf), MPI_CHAR, nextRank,
                        0, MPI_COMM_WORLD);
          if (rc != MPI_SUCCESS)
            {
              cout << "Rank " << rank
                   << " send failed, rc " << rc << endl;
              MPI_Finalize();
              exit(1);
            }
        }
    }
  cout << "Rank " << rank << " exiting normally" << endl;
  MPI_Finalize();
}
