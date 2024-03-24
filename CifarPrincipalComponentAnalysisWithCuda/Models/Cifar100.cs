using CifarPrincipalComponentAnalysis.Utilities;
using System.IO;

namespace CifarPrincipalComponentAnalysis.Models
{
    public class Cifar100 : CifarDataset
    {
        private const string Cifar100_TrainFile = "train.bin";
        private const string Cifar100_TestFile = "test.bin";
        private const string Cifar100_CoarseLabelFile = "coarse_label_names.txt";
        private const string Cifar100_FineLabelFile = "fine_label_names.txt";

        public Cifar100() : base()
        {
        }

        protected override bool CheckDirectory(string selectedPath, List<string> fileNames)
        {
            if (fileNames.Contains(Cifar100_TrainFile) == false)
            {
                CustomMaterialDesignMessageBox.Show($"Could not find {Cifar100_TrainFile} at {selectedPath}.");
                return false;
            }

            if (fileNames.Contains(Cifar100_TestFile) == false)
            {
                CustomMaterialDesignMessageBox.Show($"Could not find {Cifar100_TestFile} at {selectedPath}.");
                return false;
            }

            if (fileNames.Contains(Cifar100_CoarseLabelFile) == false)
            {
                CustomMaterialDesignMessageBox.Show($"Could not find {Cifar100_CoarseLabelFile} at {selectedPath}.");
                return false;
            }

            if (fileNames.Contains(Cifar100_FineLabelFile) == false)
            {
                CustomMaterialDesignMessageBox.Show($"Could not find {Cifar100_FineLabelFile} at {selectedPath}.");
                return false;
            }

            return true;
        }

        protected override bool ReadFiles(string selectedPath)
        {
            try
            {
                string coarseNameFile = Path.Combine(selectedPath, Cifar100_CoarseLabelFile);
                if (File.Exists(coarseNameFile) == false)
                {
                    CustomMaterialDesignMessageBox.Show($"A file {Cifar100_CoarseLabelFile} does not exist.");
                    return false;
                }

                string[] coarseNames = System.IO.File.ReadAllLines(coarseNameFile);


                string fineNameFile = Path.Combine(selectedPath, Cifar100_FineLabelFile);
                if (File.Exists(fineNameFile) == false)
                {
                    CustomMaterialDesignMessageBox.Show($"A file {Cifar100_FineLabelFile} does not exist.");
                    return false;
                }

                string[] fineNames = System.IO.File.ReadAllLines(fineNameFile);


                _dataImages = new List<SmallImageData>();
                string dataFile = Path.Combine(selectedPath, Cifar100_TrainFile);
                if (File.Exists(dataFile) == false)
                {
                    CustomMaterialDesignMessageBox.Show($"A file {dataFile} does not exist.");
                    return false;
                }

                using (FileStream fileStream = new FileStream(dataFile, FileMode.Open, FileAccess.Read))
                {
                    using (BinaryReader binaryReader = new BinaryReader(fileStream))
                    {
                        while (binaryReader.PeekChar() >= 0)
                        {
                            int label = (int) binaryReader.ReadByte();
                            string coarseName = coarseNames[label];
                            int fineLabel = (int) binaryReader.ReadByte();
                            string fineName = fineNames[fineLabel];
                            byte[] redChannelData = binaryReader.ReadBytes(ImageAreaSize);
                            byte[] greenChannelData = binaryReader.ReadBytes(ImageAreaSize);
                            byte[] blueChannelData = binaryReader.ReadBytes(ImageAreaSize);

                            _dataImages.Add(new SmallImageData(label, coarseName, fineLabel, fineName, ImageWidth, ImageHeight, redChannelData, greenChannelData, blueChannelData));
                        }
                    }
                }

                _testImages = new List<SmallImageData>();
                string testFile = Path.Combine(selectedPath, Cifar100_TestFile);
                if (File.Exists(testFile) == false)
                {
                    CustomMaterialDesignMessageBox.Show($"A file {testFile} does not exist.");
                    return false;
                }

                using (FileStream fileStream = new FileStream(testFile, FileMode.Open, FileAccess.Read))
                {
                    using (BinaryReader binaryReader = new BinaryReader(fileStream))
                    {
                        while (binaryReader.PeekChar() >= 0)
                        {
                            int label = (int) binaryReader.ReadByte();
                            string coarseName = coarseNames[label];
                            int fineLabel = (int) binaryReader.ReadByte();
                            string fineName = fineNames[fineLabel];
                            byte[] redChannelData = binaryReader.ReadBytes(ImageAreaSize);
                            byte[] greenChannelData = binaryReader.ReadBytes(ImageAreaSize);
                            byte[] blueChannelData = binaryReader.ReadBytes(ImageAreaSize);

                            _testImages.Add(new SmallImageData(label, coarseName, fineLabel, fineName, ImageWidth, ImageHeight, redChannelData, greenChannelData, blueChannelData));
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                CustomMaterialDesignMessageBox.Show($"Failed to read files in {selectedPath}. {ex}");
                return false;
            }

            return true;
        }
    }
}

