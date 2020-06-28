using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using SQLite;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection.Metadata.Ecma335;

namespace Beer.Data
{

    public class DrinkService
    {
        public static List<Drink> GetDrinks()
        {
            List<Drink> DrinksToReturn = new List<Drink>();
            SQLiteConnection drinksdb;
            if (File.Exists("Drinks.db") == false)
            {
                List<Drink> DrinksToInsert = new List<Drink>();
                using (StreamReader reader = new StreamReader(@"Assets/Products-2020-jun-27-183854.csv"))
                {
                    using (CsvReader csv = new CsvReader(reader, CultureInfo.InvariantCulture))
                    {
                        DrinksToInsert = csv.GetRecords<Drink>().ToList();
                    }
                }

                drinksdb = new SQLiteConnection("Drinks.db");
                drinksdb.CreateTable<Drink>();
                drinksdb.InsertAll(DrinksToInsert);
                drinksdb.Close();
            }

            drinksdb = new SQLiteConnection("Drinks.db");
            DrinksToReturn = drinksdb.Table<Drink>().ToList();

            drinksdb.Close();
            return DrinksToReturn;
        }

        public static string TranslateDrinkType(string drinktypeinSwedish)
        {
            string drinkinEnglish = "";
            List<DrinkTypeNames> DrinkTranslations = new List<DrinkTypeNames>();
            using (StreamReader reader = new StreamReader(@"Assets/SwedishEnglishDrinkTypes.csv"))
            {
                using (CsvReader csv = new CsvReader(reader, CultureInfo.InvariantCulture))
                {
                    DrinkTranslations = csv.GetRecords<DrinkTypeNames>().ToList();
                }
            }

            DrinkTypeNames drink = DrinkTranslations.Where(x => x.Sverige == drinktypeinSwedish).FirstOrDefault();
            if (drink != null)
            {
                drinkinEnglish = drink.English;
            }

            return drinkinEnglish;
        }

        public static string GuessDrinkType(string drinkname)
        {
            // The Drinks database must be loaded first
            if (File.Exists("Drinks.db") == false)
            {
                GetDrinks();
            }

            MLContext mlContext = new MLContext();
            DataViewSchema classifyingModelSchema;
            ITransformer trainedModel;
            if (File.Exists("DrinktypeClassifier.zip"))
            {
                trainedModel = mlContext.Model.Load("DrinktypeClassifier.zip", out classifyingModelSchema);
            }
            else
            {
                SQLiteConnection drinksdb = new SQLiteConnection("Drinks.db");
                List<DrinkType> AllDrinks = drinksdb.Table<Drink>().Select(x => new DrinkType() { Name = string.Concat(x.Namn, " ", x.Namn2).Trim(), Type = x.Varugrupp }).ToList();
                drinksdb.Close();

                IDataView trainingData = mlContext.Data.LoadFromEnumerable(AllDrinks);

                // Define features
                var dataProcessPipeline =
                    mlContext.Transforms.Conversion.MapValueToKey("Label", "PredictedLabel")
                        .Append(mlContext.Transforms.Text.FeaturizeText("Features", "Name"));

                // Use Multiclass classification
                var trainer = mlContext.MulticlassClassification.Trainers.SdcaNonCalibrated("Label", "Features");

                var trainingPipeline = dataProcessPipeline
                    .Append(trainer)
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

                // Train the model based on training data
                trainedModel = trainingPipeline.Fit(trainingData);
                mlContext.Model.Save(trainedModel, trainingData.Schema, "DrinktypeClassifier.zip");
            }
            var predFunction = mlContext.Model.CreatePredictionEngine<DrinkType, DrinkType>(trainedModel);

            var prediction = predFunction.Predict(new DrinkType() { Name = drinkname });

            return prediction.Type;
        }

        public class DrinkTypeNames
        {
            public string Sverige { get; set; }
            public string English { get; set; }
        }

        public class DrinkType
        {
            public string Name { get; set; }
            [ColumnName("PredictedLabel")]
            public string Type { get; set; }
        }

        public class Drink
        {
            public int nr { get; set; }
            public int Artikelid { get; set; }
            public int Varnummer { get; set; }
            public string Namn { get; set; }
            public string Namn2 { get; set; }
            public double Prisinklmoms { get; set; }
            public string Pant { get; set; }
            public int Volymiml { get; set; }
            public double PrisPerLiter { get; set; }
            //public DateTime Saljstart { get; set; }
            public int Utgått { get; set; }
            public string Varugrupp { get; set; }
            public string Typ { get; set; }
            public string Stil { get; set; }
            public string Forpackning { get; set; }
            public string Forslutning { get; set; }
            public string Ursprung { get; set; }
            public string Ursprunglandnamn { get; set; }
            public string Producent { get; set; }
            public string Leverantor { get; set; }
            public int? Argang { get; set; }
            public string Provadargang { get; set; }
            public string Alkoholhalt { get; set; }
            public string Sortiment { get; set; }
            public string SortimentText { get; set; }
            public int Ekologisk { get; set; }
            public int Etiskt { get; set; }
            public string EtisktEtikett { get; set; }
            public int Koscher { get; set; }
            public string RavarorBeskrivning { get; set; }
        }

        public class DrinkClassMap : ClassMap<Drink>
        {
            public DrinkClassMap()
            {
                Map(m => m.nr).Name("nr");
                Map(m => m.Artikelid).Name("Artikelid");
                Map(m => m.Varnummer).Name("Varnummer");
                Map(m => m.Namn).Name("Namn");
                Map(m => m.Namn2).Name("Namn2");
                Map(m => m.Prisinklmoms).Name("Prisinklmoms");
                Map(m => m.Pant).Name("Pant");
                Map(m => m.Volymiml).Name("Volymiml");
                Map(m => m.PrisPerLiter).Name("PrisPerLiter");
                //Map(m => m.Saljstart).Name("Saljstart").TypeConverterOption.Format("dd/MM/yyyy");
                Map(m => m.Utgått).Name("Utgått");
                Map(m => m.Varugrupp).Name("Varugrupp");
                Map(m => m.Typ).Name("Typ");
                Map(m => m.Stil).Name("Stil");
                Map(m => m.Forpackning).Name("Forpackning");
                Map(m => m.Forslutning).Name("Forslutning");
                Map(m => m.Ursprung).Name("Ursprung");
                Map(m => m.Ursprunglandnamn).Name("Ursprunglandnamn");
                Map(m => m.Producent).Name("Producent");
                Map(m => m.Leverantor).Name("Leverantor");
                Map(m => m.Argang).Name("Argang");
                Map(m => m.Provadargang).Name("Provadargang");
                Map(m => m.Alkoholhalt).Name("Alkoholhalt");
                Map(m => m.Sortiment).Name("Sortiment");
                Map(m => m.SortimentText).Name("SortimentText");
                Map(m => m.Ekologisk).Name("Ekologisk");
                Map(m => m.Etiskt).Name("Etiskt");
                Map(m => m.EtisktEtikett).Name("EtisktEtikett");
                Map(m => m.Koscher).Name("Koscher");
                Map(m => m.RavarorBeskrivning).Name("RavarorBeskrivning");
            }
        }


    }
}
