# Fraud Decision Engine in one page

## What it is

When you pay in an online shop, a fraud check runs in the background before the order is
accepted. This project builds that check: a service that looks at each purchase and, in a
few thousandths of a second, says one of three things.

| Answer | What the shop does |
|---|---|
| **Allow** | Accept the order. |
| **Review** | Ask for one extra step first, such as a code sent by text message, or have a person look at it. |
| **Deny** | Decline the order. |

## How it decides

1. **Look the purchase up.** How many different accounts have used this phone or computer
   before? How many from this internet address recently? Which country is it coming from?
2. **Apply a few certain rules.** A purchase made within a minute of creating the account
   is a bot. So is a second account on the same device within an hour, and anything from a
   sanctioned country. These are declined straight away.
3. **Score the rest.** A machine-learning model, trained on past purchases, gives a risk
   score. Above a certain level the answer is "review". The level was chosen so that the
   cost of the extra checks and the cost of the fraud that gets through are as low as
   possible together.
4. **Keep a record, and keep watching.** Every decision is stored with its reasons, and
   dashboards and 29 automatic alerts watch speed, errors and the mix of answers.

## The data

A public dataset of online purchases from 2015, the same one many fraud tutorials use.

| | |
|---|---|
| Purchases | 151,112, one per customer, from Jan 1 to Dec 16 |
| Devices | 137,956 |
| Countries (from the internet address) | 182 |
| Frauds | 14,151 (9.4%) |
| ... of which a single bot attack in January | 7,600, every one within a second of sign-up |
| Fraud rate outside that attack | 4.6%, about 1 purchase in 22 |
| Typical purchase | $35 (average fraud: $37) |

## Results

Measured on the last three and a half months of data, which the model never saw while it
was being built.

| | |
|---|---|
| Purchases in that period | 30,223 |
| Frauds among them | 1,389, worth $51,395 |
| Purchases sent to review | 2,028 (6.7%) |
| Frauds among the reviewed purchases | 469, a third of all fraud, worth $17,326 |
| How often a review is a real fraud | about 1 in 4 (23%) |
| Customers declined by the model | none: at these costs an extra check is always cheaper than turning a customer away, so declines come only from the certain rules |
| Ranking quality (PR-AUC, from 0 to 1) | 0.139; picking purchases at random scores 0.046, so about three times better than guessing |
| Overall cost: fraud that gets through plus $5 per review | $44,209, against $51,395 of fraud in the period |
| Speed, 200 purchases per second on a laptop | 6 ms typical, under 20 ms for 99 in 100. After a day idle the first half-minute was slower (99 in 100 under about 0.5 s) while data was read back into memory. |

The $5 per review and "a review stops the fraud" are assumptions. If a review cost more,
or let fraud through, the saving would shrink: the model card shows by how much.

## Seven takeaways

1. **Better information beats a cleverer method.** Four well-known machine-learning methods
   all scored the same. One new piece of information, *how many other accounts have ever
   used this device*, made all of them better at once.
2. **A more accurate model is not automatically worth more money.** The extra fraud it
   found was mixed with three times as many honest customers. Declining them all would
   have cost as much as it saved. A gentler answer, "review", is what turned the better
   model into a saving.
3. **Test with the situation your data lacks.** Nobody in this dataset buys twice, so three
   pieces of information meant something different for returning customers. One would
   have flagged loyal customers for review. They were found by trying real requests and by
   the alerts, then every input was checked, and nine were removed without any loss.
4. **Check the real thing, not a stand-in.** A model passed every test and still could not
   run, because the tests used a stand-in. Now the actual file is loaded and checked, the
   same way the live service loads it, before it can replace the running model.
5. **Know what your tools are measuring.** Two popular libraries score a very simple model
   more generously than it deserves. That made their automatic "stop training here"
   setting stop after one step. The fix was to measure with one consistent ruler.
6. **When you cannot tell if a decision was right, watch what the system does.** Nobody
   reports back which live orders were fraud. So the engine watches its own behaviour (how
   often it allows, reviews and declines, and whether incoming purchases still look like
   the ones it learned from) and raises an alert when that changes.
7. **Drop what does not earn its place.** Sex and age were inputs to the model. Measured,
   they added nothing, so they were removed: no one's answer depends on them.

## Limits

- The data is public and simplified: every customer buys only once.
- The costs are assumptions, not measurements from a real shop.
- Whether each live decision was right is not measured, because that information does not
  arrive; behaviour is watched instead.

## Read more

- [README](../README.md): how to run it, and the full results.
- [Model card](model_card.md): what the model is, how it does for different groups of
  customers, and what it must not be used for.
- [Monitoring](monitoring.md): what is watched and what each alert means.
