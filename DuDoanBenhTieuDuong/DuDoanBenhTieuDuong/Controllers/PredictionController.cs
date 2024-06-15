using System;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Newtonsoft.Json;

namespace DuDoanBenhTieuDuong.Controllers
{
    public class PredictionController : Controller
    {
        private readonly IHttpClientFactory _httpClientFactory;

        public PredictionController(IHttpClientFactory httpClientFactory)
        {
            _httpClientFactory = httpClientFactory;
        }

        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> Predict(string inputData)
        {
            try
            {
                var inputArray = inputData.Split(',').Select(double.Parse).ToArray();
                var json = JsonConvert.SerializeObject(new { input = inputArray });
                var content = new StringContent(json, Encoding.UTF8, "application/json");

                var client = _httpClientFactory.CreateClient();
                var response = await client.PostAsync("http://localhost:5000/predict", content);
                response.EnsureSuccessStatusCode();
                var responseString = await response.Content.ReadAsStringAsync();
                var prediction = JsonConvert.DeserializeObject<PredictionResult>(responseString);

                ViewBag.Prediction = prediction;
                ViewBag.InputData = inputData;
            }
            catch (Exception ex)
            {
                ModelState.AddModelError(string.Empty, $"An error occurred: {ex.Message}");
            }

            return View("Index");
        }

        public class PredictionResult
        {
            public double Prediction { get; set; }
            public string Result { get; set; }
        }
    }
}
