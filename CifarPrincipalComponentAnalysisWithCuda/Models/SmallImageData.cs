namespace CifarPrincipalComponentAnalysis.Models
{
    public class SmallImageData
    {
        private int _label;
        private string _className;

        private int _fineLabel;
        private string _fineName = "";

        private int _width;
        private int _height;

        private byte[] _redChannelData;
        public byte[] RedChannelData => _redChannelData; 

        private byte[] _greenChannelData;
        public byte[] GreenChannelData => _greenChannelData;

        private byte[] _blueChannelData;
        public byte[] BlueChannelData => _blueChannelData;

        public SmallImageData(int label, string className, int width, int height, byte[] redChannelData, byte[] greenChannelData, byte[] blueChannelData)
        {
            _label = label;
            _className = className;

            _width = width;
            _height = height;

            _redChannelData = redChannelData;
            _greenChannelData = greenChannelData;
            _blueChannelData = blueChannelData;
        }

        public SmallImageData(int label, string className, int fineLabel, string fineName, int width, int height, byte[] redChannelData, byte[] greenChannelData, byte[] blueChannelData)
          : this (label, className, width, height, redChannelData, greenChannelData, blueChannelData)
        {
            _fineLabel = fineLabel;
            _fineName = fineName;
        }
    }
}
