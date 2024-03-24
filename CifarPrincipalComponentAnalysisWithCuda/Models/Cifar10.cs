using CifarPrincipalComponentAnalysis.Utilities;
using System.IO;

namespace CifarPrincipalComponentAnalysis.Models
{
    public class Cifar10 : CifarDataset
    {
        private readonly string[] Cifar10_DataFiles =
        {
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin",
        };

        private const string Cifar10_TestFile = "test_batch.bin";
        private const string Cifar10_ClassNameFile = "batches.meta.txt";

        public Cifar10() : base()
        {
        }

        protected override bool CheckDirectory(string selectedPath, List<string> fileNames)
        {
            foreach (string file in Cifar10_DataFiles)
            {
                if (fileNames.Contains(file) == false)
                {
                    CustomMaterialDesignMessageBox.Show($"Could not find {file} at {selectedPath}.");
                    return false;
                }
            }

            if (fileNames.Contains(Cifar10_TestFile) == false)
            {
                CustomMaterialDesignMessageBox.Show($"Could not find {Cifar10_TestFile} at {selectedPath}.");
                return false;
            }

            if (fileNames.Contains(Cifar10_ClassNameFile) == false)
            {
                CustomMaterialDesignMessageBox.Show($"Could not find {Cifar10_ClassNameFile} at {selectedPath}.");
                return false;
            }

            return true;
        }

        protected override bool ReadFiles(string selectedPath)
        {
            try
            {
                string classNameFile = Path.Combine(selectedPath, Cifar10_ClassNameFile);
                if (File.Exists(classNameFile) == false)
                {
                    CustomMaterialDesignMessageBox.Show($"A file {classNameFile} does not exist.");
                    return false;
                }

                string[] classNames = System.IO.File.ReadAllLines(classNameFile);

                _dataImages = new List<SmallImageData>();
                foreach (string file in Cifar10_DataFiles)
                {
                    string  dataFile = Path.Combine(selectedPath, file);
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
                                string className = classNames[label];
                                byte[] redChannelData = binaryReader.ReadBytes(ImageAreaSize);
                                byte[] greenChannelData = binaryReader.ReadBytes(ImageAreaSize);
                                byte[] blueChannelData = binaryReader.ReadBytes(ImageAreaSize);

                                _dataImages.Add(new SmallImageData(label, className, ImageWidth, ImageHeight, redChannelData, greenChannelData, blueChannelData));
                            }
                        }
                    }
                }

                _testImages = new List<SmallImageData>();
                string testFile = Path.Combine(selectedPath, Cifar10_TestFile);
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
                            string className = classNames[label];
                            byte[] redChannelData = binaryReader.ReadBytes(ImageAreaSize);
                            byte[] greenChannelData = binaryReader.ReadBytes(ImageAreaSize);
                            byte[] blueChannelData = binaryReader.ReadBytes(ImageAreaSize);

                            _testImages.Add(new SmallImageData(label, className, ImageWidth, ImageHeight, redChannelData, greenChannelData, blueChannelData));
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
