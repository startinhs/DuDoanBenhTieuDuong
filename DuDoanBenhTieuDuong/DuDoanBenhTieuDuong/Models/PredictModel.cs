using System;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Newtonsoft.Json;

namespace DuDoanBenhTieuDuong.Models
{
    public class PredictModel : PageModel
    {
        private readonly HttpClient _httpClient;

        public PredictModel(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        [BindProperty]
        public string InputData { get; set; }
        public PredictionResult Prediction { get; set; }

        public async Task<IActionResult> OnPostAsync()
        {
            var inputArray = InputData.Split(',').Select(double.Parse).ToArray();
            var json = JsonConvert.SerializeObject(new { input = inputArray });
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync("http://localhost:5000/api/prediction/predict", content);
            var responseString = await response.Content.ReadAsStringAsync();
            Prediction = JsonConvert.DeserializeObject<PredictionResult>(responseString);

            return Page();
        }

        public class PredictionResult
        {
            public double Prediction { get; set; }
            public string Result { get; set; }
        }
    }
}
