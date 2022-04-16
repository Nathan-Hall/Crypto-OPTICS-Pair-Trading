# Crypto-OPTICS-Pair-Trading


This was a project made by Nathan Hall for fun

##### Provided under MIT License by Nathan Hall.
*Note: this library may be subtly broken or buggy. The code is released under
the MIT License – please take the following message to heart:*
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Explanation
- Implementation of Sarmento (2020) "Enhancing a Pairs Trading strategy with the application of Machine Learning"
- Gets 5-minute data of all Cryptos on Binance which have data going back over the data testing period. (This is alot of data and takes quite a while to run)
- Saves retrieved data to an excel
- Splits data in test and training datasets
- Applies PCA and OPTICS to find clusters of cryptos (cryptos that move together)
- Uses AD Fuller Test to test for cointegation, then tests for hurst exponent, half-life, mean crossings to ensure that pairs are valid

## Still to do
- Sadly this is a big project and there's still alot of work to do
- I have to re-verify the pairs in the test set, to make sure they are viable in both train and test sets
- Have to implement some type of trading system on the found data
- Have to find some way to speed up data collection process
- Most importantly, I have to clean up to code as it is quite messy (sorry)
